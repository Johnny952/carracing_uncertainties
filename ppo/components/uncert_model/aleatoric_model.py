import torch.optim as optim
import torch
from utilities import det_loss
from .basic_model import BaseTrainerModel
from models import Aleatoric
import sys
sys.path.append('../..')


class AleatoricTrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(AleatoricTrainerModel, self).__init__(nb_nets, lr,
                                                    img_stack, gamma, batch_size, buffer_capacity, device=device)
        self._model = Aleatoric(img_stack).double().to(self.device)
        self._criterion = det_loss
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)

    def forward_nograd(self, state):
        with torch.no_grad():
            (alpha, beta), (_, v, _) = self._model(state)
        return (alpha, beta), v

    def get_value_loss(self, prediction, target_v):
        _, (v, mu, log_var) = prediction
        return self._criterion(v, target_v, mu, log_var)

    def get_uncert(self, state):
        with torch.no_grad():
            (alpha, beta), (v, mu, log_var) = self._model(state)
        epistemic = torch.tensor([0])
        aleatoric = log_var
        return (alpha, beta), mu, (epistemic, aleatoric)
