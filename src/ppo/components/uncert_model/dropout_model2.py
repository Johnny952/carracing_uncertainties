import torch.optim as optim
import torch
from components.uncert_model.basic_model import BaseTrainerModel
from models.dropout import DropoutModel


class DropoutTrainerModel2(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(DropoutTrainerModel2, self).__init__(nb_nets, lr, img_stack,
                                                  gamma, batch_size, buffer_capacity, device=device)
        self.prob = 0.25
        self._model = DropoutModel(
            img_stack, prob=self.prob).double().to(self.device)
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=lr)

    def load(self, path, eval_mode=False):
        return super(DropoutTrainerModel2, self).load(path, eval_mode=False)

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

        # alpha_variance = torch.sum(torch.var(alpha_list, dim=0))
        # beta_variance = torch.sum(torch.var(beta_list, dim=0))
        epistemic = torch.std(v_list)
        aleatoric = torch.tensor([0])

        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)
