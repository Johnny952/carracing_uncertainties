import torch.optim as optim
import torch.nn.functional as F
import torch
from ppo.components.uncert_model.basic_model import BaseTrainerModel
from ppo.models.model import Net

class BootstrapTrainerModel2(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(BootstrapTrainerModel2, self).__init__(nb_nets, lr,
                                                    img_stack, gamma, batch_size, buffer_capacity, device=device)
        self._model = [Net(img_stack).double().to(
            self.device) for _ in range(nb_nets)]
        self._criterion = F.smooth_l1_loss
        self._optimizer = [optim.Adam(net.parameters(), lr=lr)
                           for net in self._model]

    def forward_nograd(self, state):
        alpha_list = []
        beta_list = []
        v_list = []
        for net in self._model:
            with torch.no_grad():
                (alpha, beta), v = net(state)
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

        # Random bagging
        # indices = [torch.utils.data.RandomSampler(range(
        #     self.buffer_capacity), num_samples=self.buffer_capacity, replacement=True) for _ in range(self.nb_nets)]
        # Random permutation
        indices = [torch.randperm(self.buffer_capacity)
                   for _ in range(self.nb_nets)]

        for _ in range(epochs):
            acc_action_loss = 0
            acc_value_loss = 0
            acc_loss = 0
            for net, optimizer, index in zip(self._model, self._optimizer, indices):
                loss, action_loss, value_loss = self.train_once(
                    net, optimizer, target_v, adv, old_a_logp, s, a, clip_param, index)
                acc_action_loss += action_loss
                acc_value_loss += value_loss
                acc_loss += loss
            self.log_loss(acc_loss, acc_action_loss, acc_value_loss)
            self._nb_update += 1

    def get_value_loss(self, prediction, target_v):
        return self._criterion(prediction[1], target_v)

    def save(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {'epoch': epoch}
        for idx, (net, optimizer) in enumerate(zip(self._model, self._optimizer)):
            tosave['model_state_dict{}'.format(idx)] = net.state_dict()
            tosave['optimizer_state_dict{}'.format(
                idx)] = optimizer.state_dict()
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        for idx in range(len(self._model)):
            self._model[idx].load_state_dict(
                checkpoint['model_state_dict{}'.format(idx)])
            self._optimizer[idx].load_state_dict(
                checkpoint['optimizer_state_dict{}'.format(idx)])
            if eval_mode:
                self._model[idx].eval()
            else:
                self._model[idx].train()
        return checkpoint['epoch']

    def get_uncert(self, state):
        alpha_list = []
        beta_list = []
        v_list = []
        for net in self._model:
            with torch.no_grad():
                (alpha, beta), v = net(state)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)

        epistemic = torch.std(v_list)
        aleatoric = torch.tensor([0])
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)
