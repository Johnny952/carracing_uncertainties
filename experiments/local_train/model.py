import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(1, 50), nn.ReLU())
        self.reg = nn.Linear(50, 1)
        self.uncert = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc(x)

        return self.reg(x), F.logsigmoid(self.uncert(x))