import torch.optim as optim
import torch
from components.uncert_model.basic_model import BaseTrainerModel
from models.dropout import DropoutModel


class DropoutTrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(DropoutTrainerModel, self).__init__(nb_nets, lr, img_stack,
                                                  gamma, batch_size, buffer_capacity, device=device)
        self.lengthscale = 0.01
        self.prob = 0.25
        self._model = DropoutModel(
            img_stack, prob=self.prob).double().to(self.device)
        # self._model.use_dropout(val=True)                          # Dropout turned off during training
        # self._criterion = smooth_l1_loss
        self.weight_decay = 1e-6
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=lr, weight_decay=self.weight_decay)

    def get_uncert(self, state):
        # self._model.use_dropout(val=False)                 # Activate dropout layers
        # Estimate uncertainties
        alpha_list = []
        beta_list = []
        v_list = []
        for _ in range(self.nb_nets):
            with torch.no_grad():
                (alpha, beta), v = self._model(state)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)
        # self._model.use_dropout(val=True)              # Deactivate dropout layers

        # alpha = torch.mean(alpha_list, dim=0)
        # beta = torch.mean(beta_list, dim=0)

        tau = self.lengthscale * (1. - self.prob) / \
            (2. * state.shape[0] * self.weight_decay)
        alpha_variance = torch.var(alpha_list, dim=0)
        beta_variance = torch.var(beta_list, dim=0)
        #epistemic = torch.mean(torch.var(alpha_list / (alpha_list + beta_list), dim=0))
        epistemic = alpha_variance + beta_variance + 1. / tau

        aleatoric = torch.tensor([0])

        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)
