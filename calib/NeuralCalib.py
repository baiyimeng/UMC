import torch
import torch.nn as nn
from utils.inputs import *


class NeuralCalib(nn.Module):
    def __init__(self, hidden_layers, feature_columns, feature_index, device, K=100):
        super(NeuralCalib, self).__init__()

        self.hidden_layers = hidden_layers
        self.device = device
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.K = K

        self.feature_num = len(feature_columns)
        self.in_dim = compute_input_dim(feature_columns)

        p = torch.logit((1.0 + torch.arange(self.K).view(1, self.K)) / (1 + self.K)).to(
            self.device
        )
        self.p = nn.Parameter(p)

        self.net = []
        hs = [self.in_dim] + hidden_layers + [1]
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

        low, high = torch.logit(torch.tensor(1 / (1 + self.K))).to(
            self.device
        ), torch.logit(torch.tensor(self.K / (1 + self.K))).to(self.device)
        y = torch.clamp(y, low, high)
        a = torch.logit((1.0 + torch.arange(self.K).view(1, self.K)) / (1 + self.K)).to(
            self.device
        )
        a = a.repeat([y.shape[0], 1])
        b = self.p
        b = b.repeat([y.shape[0], 1])
        k = (torch.searchsorted(a[0].view(-1), y.flatten(), right=True) - 1).unsqueeze(
            1
        )
        kp1 = torch.clamp(1 + k, None, self.K - 1)
        b_k = torch.gather(b, 1, k)
        b_kp1 = torch.gather(b, 1, kp1)
        a_k = torch.gather(a, 1, k)
        a_kp1 = torch.gather(a, 1, kp1)
        final_result = (
            b_k + (y - a_k) * (b_kp1 - b_k) / (a_kp1 - a_k + 1e-4) + self.net(input_all)
        )
        # final_result = b_k + (y - a_k) * (b_kp1 - b_k) / (a_kp1 - a_k + 1e-4)

        return final_result

    def compute_aux_loss(self):
        b = self.p.view(-1)
        delta_loss = [max(0.0, b[i] - b[i + 1]) for i in range(self.K - 1)]
        aux_loss = sum(delta_loss)
        return aux_loss
