from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
import torch.nn.functional as F
import torch.optim as optim
import torch
import wandb
from ppo.models.model import Net


class BaseTrainerModel:
    def __init__(
        self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device="cpu"
    ):
        self.nb_nets = nb_nets
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self._criterion = F.smooth_l1_loss

        self._model = Net(img_stack).double().to(self.device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        self._nb_update = 0
        self.variance = 0

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
            loss, action_loss, value_loss = self.train_once(
                self._model,
                self._optimizer,
                target_v,
                adv,
                old_a_logp,
                s,
                a,
                clip_param,
                sampler,
            )

            self.log_loss(loss, action_loss, value_loss)
            self._nb_update += 1

    def log_loss(self, loss, action_loss, value_loss, other_loss=None):
        to_log = {
            "Update Step": self._nb_update,
            "Loss": float(loss),
            "Action Loss": float(action_loss),
            "Value Loss": float(value_loss),
        }
        if other_loss:
            to_log["Other Loss"] = float(other_loss)
        to_log["Variance"] = float(self.variance)
        wandb.log(to_log)

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
            loss = action_loss + 2.0 * value_loss

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
            optimizer.step()

            acc_action_loss += action_loss.item()
            acc_value_loss += value_loss.item()
            acc_loss += loss.item()
            self.variance += torch.mean(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
        return acc_loss, acc_action_loss, acc_value_loss

    def get_value_loss(self, prediction, target_v):
        return self._criterion(prediction[1], target_v)

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        tosave = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if eval_mode:
            self._model.eval()
        else:
            self._model.train()
        return checkpoint["epoch"]

    def get_uncert(self, state):
        (alpha, beta), v = self.forward_nograd(state)
        epistemic = torch.mean(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
        aleatoric = torch.Tensor([0])
        return (alpha, beta), v, (epistemic, aleatoric)
