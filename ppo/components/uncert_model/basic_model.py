import sys
sys.path.append('../..')

from models import Net

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class BaseTrainerModel:
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        self.nb_nets = nb_nets
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self._criterion = F.smooth_l1_loss
        self._use_sigma = False

        self._model = Net(img_stack).double().to(self.device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)

    def forward_nograd(self, state):
        with torch.no_grad():
            (alpha, beta), v = self._model(state)[:2]
        return (alpha, beta), v

    def train(self, epochs, clip_param, database):
        (s, a, r, s_, old_a_logp) = database

        target_v = r + self.gamma * self.forward_nograd(s_)[1]
        adv = target_v - self.forward_nograd(s)[1]

        for _ in range(epochs):
            sampler = SubsetRandomSampler(range(self.buffer_capacity))
            self.train_once(self._model, self._optimizer, target_v, adv, old_a_logp, s, a, clip_param, sampler)

    def train_once(self, net, optimizer, target_v, adv, old_a_logp, s, a, clip_param, rand_sampler):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)

        for index in sampler:

            if self._use_sigma:
                (alpha, beta), v, sigma = net(s[index])
            else:
                alpha, beta = net(s[index])[0]
            dist = Beta(alpha, beta)
            a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
            ratio = torch.exp(a_logp - old_a_logp[index])

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
            action_loss = -torch.min(surr1, surr2).mean()
            if self._use_sigma:
                value_loss = self._criterion(v, target_v[index], sigma, 1.0, reduction="mean")
            else:
                value_loss = self._criterion(v, target_v[index])
            loss = action_loss + 2. * value_loss

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
            optimizer.step()

    def save(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if eval_mode:
            self._model.eval()
        else:
            self._model.train()
        return checkpoint['epoch']

    def get_uncert(self, state):
        (alpha, beta), v = self.forward_nograd(state)
        epistemic = torch.Tensor([0])
        aleatoric = torch.Tensor([0])
        return (alpha, beta), v, (epistemic, aleatoric)