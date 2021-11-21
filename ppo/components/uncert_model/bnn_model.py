import sys
sys.path.append('..')

from models import BayesianModel
from .basic_model import BaseTrainerModel

import torch
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class BNNTrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(BNNTrainerModel, self).__init__(nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device=device)
        self._model = BayesianModel(img_stack).double().to(self.device)
        self.sample_nbr = 50
        self.complexity_cost_weight = 1e-6
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)

    def forward_nograd(self, state):
        with torch.no_grad():
            alpha_list = []
            beta_list = []
            v_list = []
            for _ in range(self.nb_nets):
                (alpha, beta), v = self._model(state)
                alpha_list.append(alpha)
                beta_list.append(beta)
                v_list.append(v)
            alpha_list = torch.stack(alpha_list)
            beta_list = torch.stack(beta_list)
            v_list = torch.stack(v_list)
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0)

    def train(self, epochs, clip_param, database):
        (s, a, r, s_, old_a_logp) = database

        target_v = r + self.gamma * self.forward_nograd(s_)[1]
        adv = target_v - self.forward_nograd(s)[1]

        for _ in range(epochs):
            rand_sampler = SubsetRandomSampler(range(self.buffer_capacity))
            sampler = BatchSampler(rand_sampler, self.batch_size, False)

            for index in sampler:
                loss = 0
                for _ in range(self.sample_nbr):
                    (alpha, beta), v = self._model(s[index])
                    dist = Beta(alpha, beta)
                    a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                    ratio = torch.exp(a_logp - old_a_logp[index])

                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = self._criterion(v, target_v[index])
                    kl_loss = self._model.nn_kl_divergence() * self.complexity_cost_weight
                    
                    loss += action_loss + 2. * value_loss + kl_loss
                self._optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                self._optimizer.step()

    def get_uncert(self, state):
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

        epistemic = torch.mean(torch.var(alpha_list / (alpha_list + beta_list), dim=0))
        aleatoric = torch.Tensor([0])
        
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)