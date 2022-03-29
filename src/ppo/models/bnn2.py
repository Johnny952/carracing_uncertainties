import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn

class BayesianModel(nn.Module):
    def __init__(self, img_stack):
        super().__init__()

        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=img_stack, out_channels=8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=100),
            nn.ReLU(), 
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=1),)
        self.fc = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=100),
            nn.ReLU())
        self.alpha_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=3),
            nn.Softplus())
        self.beta_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=3),
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