import torch.nn as nn
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

@variational_estimator
class BayesianModel(nn.Module):
    def __init__(self, img_stack, n_actions):  
        super(BayesianModel, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            BayesianConv2d(img_stack, 8, (4, 4), stride=2),
            nn.LeakyReLU(),  # activation
            BayesianConv2d(8, 16, (3, 3), stride=2),  # (8, 47, 47)
            nn.LeakyReLU(),  # activation
            BayesianConv2d(16, 32, (3, 3), stride=2),  # (16, 23, 23)
            nn.LeakyReLU(),  # activation
            BayesianConv2d(32, 64, (3, 3), stride=2),  # (32, 11, 11)
            nn.LeakyReLU(),  # activation
            BayesianConv2d(64, 128, (3, 3), stride=1),  # (64, 5, 5)
            nn.LeakyReLU(),  # activation
            BayesianConv2d(128, 256, (3, 3), stride=1),  # (128, 3, 3)
            nn.LeakyReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(
            BayesianLinear(256, 100),
            nn.ReLU(),
            BayesianLinear(100, n_actions)
        )

    
    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        return self.v(x)
        # return x