import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(
        self,
        img_stack,
        steer_dim=1,
        acc_dim=2,
    ):
        super(Actor, self).__init__()
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
        
        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            )

        self.steer = nn.Sequential(
            nn.Linear(100, steer_dim),
            nn.Tanh()
            )

        self.acceleration = nn.Sequential(
            nn.Linear(100, acc_dim),
            nn.Sigmoid()
            )
        
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        steer = self.steer(x)
        acc = self.acceleration(x)
        return torch.cat((steer, acc), dim=1)
