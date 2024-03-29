import torch.nn as nn
import torch

ACTIVATION_DERIVATIVES = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    nn.functional.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) +
                            (x < 0).type(torch.FloatTensor) * -0.01,
    nn.functional.relu: lambda x: (x > 0).type(torch.FloatTensor) +
    (x < 0).type(torch.FloatTensor) * torch.exp(x)
}


class PlanarFlow(nn.Module):
    def __init__(self, D, activation=torch.tanh):
        super().__init__()
        self.D = D
        self.w = nn.Parameter(torch.empty(D))
        self.b = nn.Parameter(torch.empty(1))
        self.u = nn.Parameter(torch.empty(D))
        self.activation = activation
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, z: torch.Tensor):
        lin = (z @ self.w + self.b).unsqueeze(1)  # shape: (B, 1)
        f = z + self.u * self.activation(lin)  # shape: (B, D)
        phi = self.activation_derivative(lin) * self.w  # shape: (B, D)
        log_det = torch.log(torch.abs(1 + phi @ self.u))  # shape: (B,)

        return f, log_det


class RadialFlow(nn.Module):
    def __init__(self, D, activation=torch.tanh):
        super().__init__()

        self.z0 = nn.Parameter(torch.empty(D))
        self.log_alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        self.activation = activation
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]
        self.D = D

        nn.init.normal_(self.z0)
        nn.init.normal_(self.log_alpha)
        nn.init.normal_(self.beta)

    def forward(self, z: torch.Tensor):
        z_sub = z - self.z0
        alpha = torch.exp(self.log_alpha)
        r = torch.norm(z_sub)
        h = 1 / (alpha + r)
        f = z + self.beta * h * z_sub
        log_det = (self.D - 1) * torch.log(1 + self.beta * h) + \
            torch.log(1 + self.beta * h + self.beta -
                      self.beta * r / (alpha + r) ** 2)

        return f, log_det
