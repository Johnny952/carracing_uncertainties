import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutModel(nn.Module):
    def __init__(self, args, probs=[0.1]*7):
        super(DropoutModel, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.Dropout(p=probs[0]),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.Dropout(p=probs[1]),
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.Dropout(p=probs[2]),
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.Dropout(p=probs[3]),
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.Dropout(p=probs[4]),
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.Dropout(p=probs[5]),
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(
            nn.Linear(256, 100), 
            nn.ReLU(), 
            nn.Linear(100, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 100), 
            nn.Dropout(p=probs[-1]),
            nn.ReLU()
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(100, 3), 
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(100, 3), 
            nn.Softplus()
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(), 
            nn.Linear(100, 1),
            nn.Softplus()
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
    
    @staticmethod
    def dropout_on(m):
        if isinstance(m, nn.Dropout):
            m.train()

    @staticmethod
    def dropout_off(m):
        if isinstance(m, nn.Dropout):
            m.eval()

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        sigma = self.sigma_head(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        #sigma = self.sigma_head(x)

        return (alpha, beta), v, sigma
    
    def use_dropout(self, val=False):
        if val:
            self.apply(self.dropout_off)
        else:
            self.apply(self.dropout_on)