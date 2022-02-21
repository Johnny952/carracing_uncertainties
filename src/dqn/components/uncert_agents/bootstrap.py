import torch
import torch.nn as nn

from components.uncert_agents.abstact import AbstactAgent
from models.aleatoric import Aleatoric
from ppo.utilities.customLoss import gaussian_loss
from ppo.utilities.mixtureDist import GaussianMixture

class BootstrapAgent(AbstactAgent):
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
        super(BootstrapAgent, self).__init__(
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
        self._criterion = gaussian_loss
        self._value_scale = 1 / nb_nets

        self._model1 = [Aleatoric(img_stack, len(actions)).to(
            device) for _ in range(nb_nets)]
        self._model2 = [Aleatoric(img_stack, len(actions)).to(
            device) for _ in range(nb_nets)]

        self._optimizer1 = [torch.optim.Adam(net.parameters(), lr=self._lr)
                           for net in self._model1]
        self._optimizer2 = [torch.optim.Adam(net.parameters(), lr=self._lr)
                           for net in self._model2]
    
    def get_values(self, observation):
        values_list = []
        log_var_list = []
        for model in self._model1:
            _, values, log_var = model(observation)
            values_list.append(values)
            log_var_list.append(log_var)
        values_list = torch.stack(values_list).squeeze(dim=1)
        sigma_list = torch.exp(torch.stack(log_var_list)).squeeze(dim=1)
        distribution = GaussianMixture(values_list, sigma_list, device=self._device)
        values = distribution.mean.unsqueeze(dim=0)
        _, index = torch.max(values, dim=-1)
        epistemic = distribution.std[index[0].cpu().numpy()]
        aleatoric = torch.Tensor([0])
        return index, epistemic, aleatoric

    def update(self):
        states, actions, next_states, rewards, dones = self.unpack(
            self._buffer.sample()
        )
        # Random bagging
        indices = [torch.utils.data.RandomSampler(range(
            self._buffer.batch_size), num_samples=self._buffer.batch_size, replacement=True) for _ in range(self.nb_nets)]
        # Random permutation
        # indices = [torch.randperm(self._buffer.batch_size)
        #            for _ in range(self.nb_nets)]
        acc_loss1 = 0
        acc_loss2 = 0
        for model1, model2, optimizer1, optimizer2, index in zip(self._model1, self._model2, self._optimizer1, self._optimizer2, indices):
            loss1, loss2 = self.compute_loss(model1, model2, states[index], actions[index], next_states[index], rewards[index], dones[index])

            optimizer1.zero_grad()
            loss1.backward()
            if self._clip_grad:
                for param in model1.parameters():
                    param.grad.data.clamp_(-1, 1)
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            if self._clip_grad:
                for param in model2.parameters():
                    param.grad.data.clamp_(-1, 1)
            optimizer2.step()

            acc_loss1 += loss1.item()
            acc_loss2 += loss2.item()

        self.log_loss(acc_loss1, acc_loss2)
        self._nb_update += 1
    
    def compute_loss(self, model1, model2, states, actions, next_states, rewards, dones):
        _, values1, log_sigma1 = model1(states)
        _, values2, log_sigma2 = model2(states)
        
        curr_Q1 = values1.gather(1, actions).squeeze(dim=-1)
        curr_Q2 = values2.gather(1, actions).squeeze(dim=-1)

        curr_log_sigma1 = log_sigma1.gather(1, actions).squeeze(dim=-1)
        curr_log_sigma2 = log_sigma2.gather(1, actions).squeeze(dim=-1)

        next_Q = torch.min(
            torch.max(model1(next_states)[1], 1)[0],
            torch.max(model2(next_states)[1], 1)[0],
        ).squeeze(dim=-1)
        expected_Q = rewards + (1 - dones) * self._gamma * next_Q

        loss1 = self._criterion(curr_Q1, expected_Q.detach(), curr_log_sigma1)
        loss2 = self._criterion(curr_Q2, expected_Q.detach(), curr_log_sigma2)

        return loss1, loss2
    
    def save_param(self, epoch, path="param/ppo_net_param.pkl"):
        tosave = {
            "epoch": epoch,
        }
        for idx, (model1, model2, optimizer1, optimizer2) in enumerate(zip(self._model1, self._model2, self._optimizer1, self._optimizer2)):
            tosave[f"model1_state_dict_{idx}"] = model1
            tosave[f"model2_state_dict_{idx}"] = model2
            tosave[f"optimizer1_state_dict_{idx}"] = optimizer1
            tosave[f"optimizer2_state_dict_{idx}"] = optimizer2
        torch.save(tosave, path)

    def load_param(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        for idx, (model1, model2, optimizer1, optimizer2) in enumerate(zip(self._model1, self._model2, self._optimizer1, self._optimizer2)):
            model1.load_state_dict(checkpoint[f"model1_state_dict_{idx}"])
            model2.load_state_dict(checkpoint[f"model2_state_dict_{idx}"])
            optimizer1.load_state_dict(checkpoint[f"optimizer1_state_dict_{idx}"])
            optimizer2.load_state_dict(checkpoint[f"optimizer2_state_dict_{idx}"])

            if eval_mode:
                model1.eval()
                model2.eval()
            else:
                model1.train()
                model2.train()
        return checkpoint["epoch"]