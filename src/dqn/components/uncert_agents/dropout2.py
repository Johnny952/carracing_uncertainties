import torch
import torch.nn as nn

from components.uncert_agents.abstact import AbstactAgent
from models.dropout import Dropout

class DropoutAgent2(AbstactAgent):
    def __init__(
        self,
        nb_nets,
        img_stack,
        actions,
        learning_rate,
        gamma,
        buffer,
        epsilon,
        batch_size,
        device="cpu",
        clip_grad=False,
    ):
        super(DropoutAgent2, self).__init__(
            nb_nets,
            img_stack,
            actions,
            learning_rate,
            gamma,
            buffer,
            epsilon,
            batch_size,
            device=device,
            clip_grad=clip_grad,
        )

        self.nb_nets = nb_nets
        self.prob = 0.25
        self._criterion = nn.MSELoss()

        self._model1 = Dropout(img_stack, len(actions)).to(self._device)
        self._model2 = Dropout(img_stack, len(actions)).to(self._device)

        self._optimizer1 = torch.optim.Adam(self._model1.parameters(), lr=self._lr)
        self._optimizer2 = torch.optim.Adam(self._model2.parameters(), lr=self._lr)
    
    def get_values(self, observation):
        values_list = []
        for _ in range(self.nb_nets):
            values = self._model1(observation)
            values_list.append(values)
        values_list = torch.stack(values_list)
        values = torch.mean(values_list, dim=0)
        _, index = torch.max(values, dim=-1)
        epistemic = torch.sum(torch.var(values_list, dim=0))
        aleatoric = torch.Tensor([0])
        return index, epistemic, aleatoric
    
    def load_param(self, path, eval_mode=False):
        return super(DropoutAgent2, self).load_param(path, eval_mode=False)