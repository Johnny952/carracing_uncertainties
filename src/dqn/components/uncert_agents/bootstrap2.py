import torch
import wandb
import numpy as np
import torch.nn as nn

from components.uncert_agents.abstact import AbstactAgent
from models.base import Net

class BootstrapAgent2(AbstactAgent):
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
        super(BootstrapAgent2, self).__init__(
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
        self._criterion = nn.MSELoss()

        self._model1 = [Net(img_stack, len(actions)).to(
            device) for _ in range(nb_nets)]
        self._model2 = [Net(img_stack, len(actions)).to(
            device) for _ in range(nb_nets)]

        self._optimizer1 = [torch.optim.Adam(net.parameters(), lr=self._lr)
                           for net in self._model1]
        self._optimizer2 = [torch.optim.Adam(net.parameters(), lr=self._lr)
                           for net in self._model2]
    
    def get_values(self, observation):
        values_list = []
        for model in self._model1:
            v = model(observation)
            values_list.append(v)
        values_list = torch.stack(values_list).squeeze(dim=1)
        wandb.log({
            "Magnitude Value": torch.abs(values_list).sum()
        })
        values = torch.mean(values_list, dim=0)
        _, index = torch.max(values, dim=-1)
        epistemic = torch.sum(torch.var(values_list, dim=0))
        aleatoric = torch.Tensor([0])
        return index, epistemic, aleatoric

    def update(self):
        states, actions, next_states, rewards, dones = self.unpack(
            self._buffer.sample()
        )
        # Random bagging
        # indices = [np.random.choice(
        #     np.array(range(self._buffer.batch_size)),
        #     size=self._buffer.batch_size, replace=True) for _ in range(self.nb_nets)]
        # Random permutation
        indices = [torch.randperm(self._buffer.batch_size)
                   for _ in range(self.nb_nets)]
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
        curr_Q1 = model1(states).gather(1, actions).squeeze(dim=-1)
        curr_Q2 = model2(states).gather(1, actions).squeeze(dim=-1)

        next_Q = torch.min(
            torch.max(model1(next_states)[1], 1)[0],
            torch.max(model2(next_states)[1], 1)[0],
        ).squeeze(dim=-1)
        expected_Q = rewards + (1 - dones) * self._gamma * next_Q

        loss1 = self._criterion(curr_Q1, expected_Q.detach())
        loss2 = self._criterion(curr_Q2, expected_Q.detach())

        return loss1, loss2
    
    def save_param(self, epoch, path="param/ppo_net_param.pkl"):
        tosave = {
            "epoch": epoch,
        }
        for idx, (model1, model2, optimizer1, optimizer2) in enumerate(zip(self._model1, self._model2, self._optimizer1, self._optimizer2)):
            tosave[f"model1_state_dict_{idx}"] = model1.state_dict(),
            tosave[f"model2_state_dict_{idx}"] = model2.state_dict(),
            tosave[f"optimizer1_state_dict_{idx}"] = optimizer1.state_dict(),
            tosave[f"optimizer2_state_dict_{idx}"] = optimizer2.state_dict(),
        torch.save(tosave, path)

    def load_param(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        for idx in range(self.nb_nets):
            self._model1[idx].load_state_dict(checkpoint[f"model1_state_dict_{idx}"][0])
            self._model2[idx].load_state_dict(checkpoint[f"model2_state_dict_{idx}"][0])
            self._optimizer1[idx].load_state_dict(checkpoint[f"optimizer1_state_dict_{idx}"][0])
            self._optimizer2[idx].load_state_dict(checkpoint[f"optimizer2_state_dict_{idx}"][0])
            if eval_mode:
                self._model1[idx].eval()
                self._model2[idx].eval()
            else:
                self._model1[idx].train()
                self._model2[idx].train()
        return checkpoint["epoch"]