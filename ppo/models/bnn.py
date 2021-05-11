import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

@variational_estimator
class BayesianModel(nn.Module):
    def __init__(self, img_stack):
        super().__init__()

        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            BayesianConv2d(img_stack, 8, (4, 4), stride=2),
            nn.ReLU(),  # activation
            BayesianConv2d(8, 16, (3, 3), stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            BayesianConv2d(16, 32, (3, 3), stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            BayesianConv2d(32, 64, (3, 3), stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            BayesianConv2d(64, 128, (3, 3), stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            BayesianConv2d(128, 256, (3, 3), stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(
            BayesianLinear(256, 100), 
            nn.ReLU(), 
            BayesianLinear(100, 1))
        self.fc = nn.Sequential(
            BayesianLinear(256, 100), 
            nn.ReLU())
        self.alpha_head = nn.Sequential(
            BayesianLinear(100, 3), 
            nn.Softplus())
        self.beta_head = nn.Sequential(
            BayesianLinear(100, 3), 
            nn.Softplus())

        #self.apply(self._weights_init)
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        
        return (alpha, beta), v