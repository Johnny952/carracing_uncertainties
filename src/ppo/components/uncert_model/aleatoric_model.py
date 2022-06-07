import torch.optim as optim
import torch
from ppo.utilities.customLoss import det_loss, det_loss2
from ppo.components.uncert_model.basic_model import BaseTrainerModel
from ppo.models.aleatoric import Aleatoric

from torch.utils.data.sampler import BatchSampler
from torch.distributions import Beta


class AleatoricTrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(AleatoricTrainerModel, self).__init__(nb_nets, lr,
                                                    img_stack, gamma, batch_size, buffer_capacity, device=device)
        self._model = Aleatoric(img_stack).double().to(self.device)
        self._criterion = det_loss
        self._weight_decay = 1e-30
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)#, weight_decay=self._weight_decay

    def forward_nograd(self, state):
        with torch.no_grad():
            (alpha, beta), (_, v, _) = self._model(state)
        return (alpha, beta), v

    def get_value_loss(self, prediction, target_v):
        _, (v, mu, log_var) = prediction
        return self._criterion(v, target_v, mu, log_var, weight_decay=self._weight_decay)

    def get_uncert(self, state):
        with torch.no_grad():
            (alpha, beta), (v, mu, log_var) = self._model(state)
        epistemic = torch.tensor([0])
        aleatoric = log_var
        return (alpha, beta), mu, (epistemic, aleatoric)

    def train_once(
        self, net, optimizer, target_v, adv, old_a_logp, s, a, clip_param, rand_sampler
    ):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)
        acc_action_loss = 0
        acc_value_loss = 0
        acc_loss = 0
        self.get_value_lossvariance = 0

        for index in sampler:

            prediction = net(s[index])
            alpha, beta = prediction[0]
            dist = Beta(alpha, beta)
            a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
            ratio = torch.exp(a_logp - old_a_logp[index])

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.get_value_loss(prediction, target_v[index])
            loss = action_loss + 2.0 * value_loss / 500

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
            optimizer.step()

            acc_action_loss += action_loss.item()
            acc_value_loss += value_loss.item()
            acc_loss += loss.item()
            self.variance += torch.mean(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
        return acc_loss, acc_action_loss, acc_value_loss
