import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, img_stack, n_actions):
        """Net Constructor

        Args:
            img_stack (int): Number of stacked images
            n_actions (int): Number of actions or outputs of the model
        """        
        super(Net, self).__init__()
        # self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
        #     nn.Conv2d(img_stack, 16, kernel_size=5), # (16, 92, 92)
        #     nn.ReLU(),  # activation
        #     nn.MaxPool2d(6, stride=6), # (16, 15, 15)
        #     nn.Conv2d(16, 64, kernel_size=3),  # (64, 13, 13)
        #     nn.ReLU(),  # activation
        #     nn.MaxPool2d(6, stride=6),        # (64, 2, 2)
        #     nn.Conv2d(64, 256, kernel_size=2),  # (256, 1, 1)
        #     nn.ReLU(),  # activation
        #     )
        # self.cnn_base = nn.Sequential(
        #     nn.Conv2d(img_stack, 6, 7),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(6),
        #     nn.Conv2d(6, 12, 4),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(12),
        #     nn.Flatten(1),
        #     nn.Linear(5292, 216),
        #     nn.ReLU(),
        #     nn.Linear(216, n_actions)
        # )
        self.cnn_base = nn.Sequential(
            nn.Conv2d(img_stack, 6, kernel_size=7, stride=3),       # (N, img_stack, 96, 96) => (N, 6, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # (N, 6, 30, 30) => (N, 6, 15, 15)

            nn.Conv2d(6, 12, kernel_size=4),                        # (N, 6, 15, 15) => (N, 12, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # (N, 12, 12, 12) => (N, 12, 6, 6)
            nn.Flatten(),                                           # (N, 12, 6, 6) => (N, 432)
            nn.Linear(432, 216),
            nn.ReLU(),
            nn.Linear(216, n_actions),
        )
        # self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
        #     nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
        #     nn.ReLU(),  # activation
        # )  # output shape (256, 1, 1)
        # self.v = nn.Sequential(
        #     nn.Linear(256, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, n_actions)
        # )
        self.apply(self._weights_init)
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        x = self.cnn_base(x)
        # x = x.view(-1, 256)
        # return self.v(x)
        return x


if __name__ == "__main__":
    model = Net(4, 10)

    obs = torch.randn((1, 4, 96, 96), dtype=torch.float)
    print(model(obs))