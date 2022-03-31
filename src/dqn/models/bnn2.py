import torch.nn as nn
import torchbnn as bnn

class BayesianModel(nn.Module):
    def __init__(self, img_stack, n_actions):  
        super(BayesianModel, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=img_stack, out_channels=8, kernel_size=4, stride=2),
            nn.LeakyReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU(),  # activation
            bnn.BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=100),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=n_actions),
        )

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        return self.v(x)
        # return x