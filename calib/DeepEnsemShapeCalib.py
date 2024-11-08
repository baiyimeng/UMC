import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.inputs import *


class DESC(nn.Module):
    def __init__(self, hidden_layers, feature_columns, feature_index, device, K=100):
        super(DESC, self).__init__()
        self.bucket_embedding = nn.Embedding(K, 16).to(device)

        self.hidden_layers = hidden_layers
        self.device = device
        self.feature_columns = feature_columns
        self.feature_index = feature_index

        self.field_num = len(feature_columns)
        self.in_dim = compute_input_dim(feature_columns)
        self.K = 100

        self.net1 = [
            nn.Linear(self.in_dim // self.field_num * (self.field_num + 1), 64),
            nn.ReLU(),
        ]
        self.net1 = nn.Sequential(*self.net1).to(device)

        self.net2 = [
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        ]
        self.net2 = nn.Sequential(*self.net2).to(device)

        self.net3 = [
            nn.Linear(64, len(self.feature_columns)),
            nn.Softmax(dim=1),
        ]
        self.net3 = nn.Sequential(*self.net3).to(device)

        single_list = []
        for i in range(self.field_num):
            single_list.append(
                SFSC(
                    [self.in_dim // self.field_num * 3] + self.hidden_layers + [48],
                    self.device,
                    i,
                )
            )
        self.single = nn.ModuleList(single_list)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=1e-4)

    def forward(self, x, y, embedding_dict):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        bucket_id = (F.sigmoid(y) * self.K).int()
        bucket_id = torch.clamp(bucket_id, None, self.K - 1)
        y_embedding = self.bucket_embedding(bucket_id)
        s_embedding = torch.cat(sparse_embedding_list, dim=1)

        # stop gradient
        s_embedding = s_embedding.detach()
        y = y.detach()

        input_all_list = torch.cat([s_embedding, y_embedding], dim=1)
        input_all = torch.flatten(input_all_list, start_dim=1)
        value_temp = self.net1(input_all)
        value_output = self.net2(value_temp)
        global_weight = self.net3(value_temp)

        shape_list = []
        for shape_model in self.single:
            shape_list.append(shape_model(input_all_list, y))
        shape_tensor = torch.cat(shape_list, dim=1)

        final_result = torch.logit(
            torch.sum(global_weight * shape_tensor, dim=1, keepdim=True) * value_output
        )
        return final_result


class SFSC(nn.Module):
    def __init__(self, hidden_layers, device, field_index):
        super(SFSC, self).__init__()
        self.net = []
        hs = hidden_layers
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        self.net = self.net.to(device)
        self.device = device
        self.field_index = field_index

    def forward(self, x, y):
        y_embedding = x[:, -1, :]
        f_embedding = x[:, :-1, :]
        t_embedding = f_embedding[:, self.field_index, :]
        x_embedding = torch.stack(
            [
                f_embedding[:, i, :]
                for i in range(f_embedding.shape[1])
                if i != self.field_index
            ],
            dim=1,
        )

        attention_weight = F.softmax(
            torch.matmul(x_embedding, t_embedding.unsqueeze(-1))
            / (t_embedding.shape[-1] ** 0.5),
            dim=1,
        )
        x_embedding = torch.sum(attention_weight * x_embedding, dim=1)

        final_input = torch.cat([y_embedding, t_embedding, x_embedding], dim=1)

        shape_weight = self.net(final_input)  # B * 48
        shape_weight = F.softmax(shape_weight, dim=1)

        a = torch.tensor(np.array(list(range(5, 21))) / 10).to(self.device)

        def pow(y, a):
            return torch.pow(F.sigmoid(y), a)

        def scaling(y, a):
            return F.sigmoid(y * a)

        def log(y, a):
            return torch.log(1 + F.sigmoid(y) * a) / torch.log(1 + a)

        ens = torch.concat([pow(y, a), scaling(y, a), log(y, a)], dim=1)
        res = torch.sum(ens * shape_weight, dim=1, keepdim=True)
        return res.float()
