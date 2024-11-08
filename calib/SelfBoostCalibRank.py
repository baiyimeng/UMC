import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.inputs import *


class SBCR(nn.Module):
    def __init__(self, hidden_layers, feature_columns, feature_index, device, K=100):
        super(SBCR, self).__init__()

        self.hidden_layers = hidden_layers
        self.device = device
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.K = K

        self.in_dim = compute_input_dim(feature_columns)

        self.net = []
        hs = [self.in_dim] + hidden_layers + [self.K]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net).to(device)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y, embedding_dict):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        s_embedding = torch.cat(sparse_embedding_list, dim=1)
        input_all_list = s_embedding

        # stop gradient
        input_all = torch.flatten(input_all_list, start_dim=1).detach()
        y = y.detach()

        y = F.sigmoid(y)
        a = F.softmax(self.net(input_all), dim=1)
        b = torch.cat([torch.zeros_like(y), a.cumsum(dim=1)], dim=1)
        k = (y * self.K).to(torch.int64)
        kp1 = torch.clamp(1 + k, 0, self.K).to(torch.int64)
        b_k = torch.gather(b, 1, k)
        b_kp1 = torch.gather(b, 1, kp1)
        final_result = torch.clamp(
            b_k + (self.K * y - k) * (b_kp1 - b_k), 1e-4, 1 - 1e-4
        )
        final_result = torch.logit(final_result)

        return final_result
