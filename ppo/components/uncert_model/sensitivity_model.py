import sys
sys.path.append('..')

from models import Sensitivity
from .basic_model import BaseTrainerModel

import torch
import torch.optim as optim

class SensitivityTrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(SensitivityTrainerModel, self).__init__(nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device=device)
        self._model = Sensitivity(img_stack).double().to(self.device)
        self.input_range = [0, 255]
        self.factor = 1/255
        self.delta = self.factor * (self.input_range[1] - self.input_range[0])
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)

    def get_uncert(self, state):
        # Random matrix -1/0/1
        rand_dir = self.delta*(torch.empty(self.nb_nets, state.shape[1], state.shape[2], state.shape[3]).random_(3).double().to(self.device) - 1)
        rand_dir += state
        rand_dir[rand_dir > self.input_range[1]] = self.input_range[1]
        rand_dir[rand_dir < self.input_range[0]] = self.input_range[0]

        # Estimate uncertainties
        with torch.no_grad():
            (alpha, beta), v = self._model(rand_dir)

        #var = alpha*beta / ((alpha+beta+1)*(alpha+beta)**2)
        epistemic = torch.mean(torch.var(alpha / (alpha + beta), dim=0))
        aleatoric = torch.tensor([0])

        # Predict
        with torch.no_grad():
            (alpha, beta), v = self._model(state)
        # aleatoric = sigma

        #return (torch.mean(alpha, dim=0).view(1, -1), torch.mean(beta, dim=0).view(1, -1)), torch.mean(v, dim=0), (epistemic, aleatoric)
        return (alpha, beta), v, (epistemic, aleatoric)
