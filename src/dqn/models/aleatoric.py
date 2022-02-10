import torch.nn as nn
import torch

class Aleatoric(nn.Module):
    def __init__(self, img_stack, n_actions):
        """Net Constructor

        Args:
            img_stack (int): Number of stacked images
            n_actions (int): Number of actions or outputs of the model
        """        
        super(Aleatoric, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(),  # activation
            # nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),  # activation
            # nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),  # activation
            # nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # activation
            # nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(),  # activation
            # nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(),  # activation
            # nn.ReLU(),
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions)
        )
        self.log_var = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )
        self.apply(self._weights_init)
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def reparameterize(self, mu, log_var):        
        sigma = torch.exp(0.5 * log_var) + 1e-5
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
    
    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        mu = self.v(x)
        log_var = self.log_var(x)
        reparametrization = self.reparameterize(mu, log_var)
        return reparametrization, mu, log_var
