import torch.nn as nn

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, img_stack):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
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
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
    
    def partial_forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)

        v = self.v(x)
        y = self.fc(x)
        alpha = self.alpha_head(y) + 1
        beta = self.beta_head(y) + 1

        return x, (alpha, beta), v

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

class NoiseNet(Net):
    def __init__(self, img_stack):
        super(NoiseNet, self).__init__(img_stack)
        self._sigma_head = nn.Sequential(
            nn.Linear(256, 100), 
            nn.ReLU(),
            nn.Linear(100, 1), 
            nn.Softplus()
            )
        super(NoiseNet, self).apply(self._weights_init)
    
    def forward(self, x):
        x, (alpha, beta), v = super(NoiseNet, self).partial_forward(x)
        
        sigma = self._sigma_head(x)
        return (alpha, beta), v, sigma