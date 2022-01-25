import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
import math
from utilities.normalizing_flows import PlanarFlow, RadialFlow

# https://towardsdatascience.com/variational-inference-with-normalizing-flows-on-mnist-9258bbcf8810
class FCNEncoder(nn.Module):
    def __init__(self, img_stack, output_dim=64):
        super().__init__()

        self._cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

        self._mu_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self._log_sigma_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self._cnn_base(x)
        x = torch.flatten(x, start_dim=1)
        return self._mu_layers(x), self._log_sigma_layers(x)


class FlowModel(nn.Module):
    def __init__(self, flows: 'list(str)', D: int, activation=torch.tanh, device='cpu'):
        super().__init__()
        self.device = device
        self.prior = MultivariateNormal(torch.zeros(D), torch.eye(D))
        self.net = []
        for i in range(len(flows)):
            layer_class = eval(flows[i])
            self.net.append(layer_class(D, activation))
        self.net = nn.Sequential(*self.net)
        self.D = D
        self.log_pi = torch.log(torch.tensor(2 * math.pi).to(self.device))

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        mu: tensor with shape (batch_size, D)
        sigma: tensor with shape (batch_size, D)
        """
        sigma = torch.exp(log_sigma)
        batch_size = mu.shape[0]
        samples = self.prior.sample(torch.Size([batch_size])).to(self.device)
        z = samples * sigma + mu

        z0 = z.clone().detach()
        # log_prob_z0 = torch.sum(
        #     -0.5 * torch.log(torch.tensor(2 * math.pi)) -
        #     log_sigma - 0.5 * ((z - mu) / sigma) ** 2,
        #     axis=1)
        log_prob_z0 = torch.sum(
            -0.5 * self.log_pi -
            log_sigma - 0.5 * ((z - mu) / sigma) ** 2,
            axis=1)

        log_det = torch.zeros((batch_size,)).to(self.device)

        for layer in self.net:
            z, ld = layer(z)
            log_det += ld

        # log_prob_zk = torch.sum(
        #     -0.5 * (torch.log(torch.tensor(2 * math.pi)) + z ** 2),
        #     axis=1)
        log_prob_zk = torch.sum(
            -0.5 * (self.log_pi + z ** 2),
            axis=1)

        return z, log_prob_z0, log_prob_zk, log_det

class FCNDecoder(nn.Module):
    def __init__(self, img_stack, input_dim=64):
        super().__init__()

        self._mlp_base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self._cnn_layers = nn.Sequential(
            # (256, 1, 1) => (128, 3, 3)
            nn.ConvTranspose2d(256, 128, 3, stride=1),
            nn.ReLU(),
            # (128, 3, 3) => (64, 5, 5)
            nn.ConvTranspose2d(128, 64, 3, stride=1),
            nn.ReLU(),
            # (64, 5, 5) => (32, 11, 11)
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(),
            # (32, 11, 11) => (16, 23, 23)
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            nn.ReLU(),
            # (16, 23, 23) => (8, 47, 47)
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.ReLU(),
            # (8, 47, 47) => (4, 96, 96)
            nn.ConvTranspose2d(8, img_stack, 4, stride=2),
        )

    def forward(self, z: torch.Tensor):
        z = self._mlp_base(z)
        z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return self._cnn_layers(z)
