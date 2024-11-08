import torch
import torch.nn as nn
from calib.ParallelNeuralIntegral import ParallelNeuralIntegral
from utils.inputs import *


class IntegrandNN(nn.Module):
    def __init__(self, in_dim, hidden_layers, device):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_dim] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())

        self.net = nn.Sequential(*self.net)
        self.net = self.net.to(device)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.0


class UMC(nn.Module):
    def __init__(
        self,
        hidden_layers,
        feature_columns,
        feature_index,
        device,
        nb_steps=50,
        rescaling=True,
    ):
        super(UMC, self).__init__()
        in_dim = 1 + compute_input_dim(feature_columns)
        self.integrand = IntegrandNN(in_dim, hidden_layers, device=device)
        self.rescaling = rescaling

        b = torch.zeros(1).to(device)
        self.bias = nn.Parameter(b)

        in_dim = compute_input_dim(feature_columns)

        self.net = []
        hs = [in_dim] + [200, 200] + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net).to(device)

        self.device = device
        self.nb_steps = nb_steps
        self.feature_columns = feature_columns
        self.feature_index = feature_index

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, x, y, embedding_dict):

        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        y0 = torch.zeros_like(y).to(self.device)
        h_out = torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1)
        h_in = h_out

        # stop gradient
        h_out = h_out.detach()
        h_in = h_in.detach()
        y = y.detach()

        offset = self.bias
        weight = torch.exp(self.net(h_out)[:, [0]])
        bias = self.net(h_out)[:, [1]]

        result = (
            ParallelNeuralIntegral.apply(
                y0,
                y,
                self.integrand,
                flatten(self.integrand.parameters()),
                h_in,
                self.nb_steps,
            )
            + offset
        )
        if self.rescaling:
            result = weight * result + bias

        return result


class UMNN(nn.Module):
    def __init__(self, hidden_layers, device, nb_steps=50):
        super(UMNN, self).__init__()
        in_dim = 2
        self.integrand = IntegrandNN(in_dim, hidden_layers, device=device)

        b = torch.zeros(1).to(device)
        self.bias = nn.Parameter(b)

        self.device = device
        self.nb_steps = nb_steps

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, y):
        y0 = torch.zeros_like(y).to(self.device)
        h = torch.zeros_like(y).to(self.device)

        # stop gradient
        h = h.detach()
        y = y.detach()

        offset = self.bias
        result = (
            ParallelNeuralIntegral.apply(
                y0,
                y,
                self.integrand,
                flatten(self.integrand.parameters()),
                h,
                self.nb_steps,
            )
            + offset
        )
        return result
