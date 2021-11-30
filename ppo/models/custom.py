import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
import math

# https://towardsdatascience.com/variational-inference-with-normalizing-flows-on-mnist-9258bbcf8810
# TODO: Convert encoder and decoder for images

class FCNEncoder(nn.Module):
    def __init__(self, hidden_sizes: 'list(int)', dim_input: int, activation=nn.ReLU()):
        super().__init__()
        hidden_sizes = [dim_input] + hidden_sizes
        self.net = []
        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(nn.ReLU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class FlowModel(nn.Module):
    def __init__(self, flows: 'list(str)', D: int, activation=torch.tanh):
        super().__init__()
        self.prior = MultivariateNormal(torch.zeros(D), torch.eye(D))
        self.net = []
        for i in range(len(flows)):
            layer_class = eval(flows[i])
            self.net.append(layer_class(D, activation))
        self.net = nn.Sequential(*self.net)
        self.D = D

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        mu: tensor with shape (batch_size, D)
        sigma: tensor with shape (batch_size, D)
        """
        sigma = torch.exp(log_sigma)
        batch_size = mu.shape[0]
        samples = self.prior.sample(torch.Size([batch_size]))
        z = samples * sigma + mu

        z0 = z.clone().detach()
        log_prob_z0 = torch.sum(
            -0.5 * torch.log(torch.tensor(2 * math.pi)) -
            log_sigma - 0.5 * ((z - mu) / sigma) ** 2,
            axis=1)

        log_det = torch.zeros((batch_size,))

        for layer in self.net:
            z, ld = layer(z)
            log_det += ld

        log_prob_zk = torch.sum(
            -0.5 * (torch.log(torch.tensor(2 * math.pi)) + z ** 2),
            axis=1)

        return z, log_prob_z0, log_prob_zk, log_det

# TODO: Convert encoder and decoder for images


class FCNDecoder(nn.Module):
    def __init__(self, hidden_sizes: 'list(int)', dim_input: int, activation=nn.ReLU):
        super().__init__()
        hidden_sizes = [dim_input] + hidden_sizes
        self.net = []
        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(activation())
        self.net = nn.Sequential(*self.net)

    def forward(self, z: torch.Tensor):
        return self.net(z)
