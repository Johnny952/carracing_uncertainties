import torch
import torch.nn as nn

from components.uncert_agents.abstact import AbstactAgent
from models.base import Net

class SensitivityAgent(AbstactAgent):
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
        super(SensitivityAgent, self).__init__(
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
        self.input_range = [-1, 1]
        self._noise_variance = 0.1
        self._criterion = nn.MSELoss()

        self._model1 = Net(img_stack, len(actions)).to(self._device)
        self._model2 = Net(img_stack, len(actions)).to(self._device)

        self._optimizer1 = torch.optim.Adam(self._model1.parameters(), lr=self._lr)
        self._optimizer2 = torch.optim.Adam(self._model2.parameters(), lr=self._lr)
    
    def get_values(self, observation):
        values = self._model1(observation)
        _, index = torch.max(values, dim=-1)

        # Random matrix
        size = (self.nb_nets, observation.shape[1], observation.shape[2], observation.shape[3])
        rand_dir = torch.normal(
            torch.zeros(size), self._noise_variance*torch.ones(size)
        ).float().to(self._device)
        rand_dir += observation
        rand_dir[rand_dir > self.input_range[1]] = self.input_range[1]
        rand_dir[rand_dir < self.input_range[0]] = self.input_range[0]
        rand_values = self._model1(rand_dir)

        epistemic = torch.mean(torch.var(rand_values, dim=0))
        aleatoric = torch.Tensor([0])
        return index, epistemic, aleatoric
    
    def compute_loss(self, states, actions, next_states, rewards, dones):
        curr_Q1 = self._model1(states).gather(1, actions).squeeze(dim=-1)
        curr_Q2 = self._model2(states).gather(1, actions).squeeze(dim=-1)

        next_Q = torch.min(
            torch.max(self._model1(next_states), 1)[0],
            torch.max(self._model2(next_states), 1)[0],
        ).squeeze(dim=-1)
        expected_Q = rewards + (1 - dones) * self._gamma * next_Q

        loss1 = self._criterion(curr_Q1, expected_Q.detach())
        loss2 = self._criterion(curr_Q2, expected_Q.detach())

        return loss1, loss2
    
    def save_param(self, epoch, path="param/ppo_net_param.pkl"):
        tosave = {
            "epoch": epoch,
            "model1_state_disct": self._model1.state_dict(),
            "model2_state_disct": self._model2.state_dict(),
            "optimizer1_state_dict": self._optimizer1.state_dict(),
            "optimizer2_state_dict": self._optimizer2.state_dict(),
        }
        torch.save(tosave, path)

    def load_param(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model1.load_state_dict(checkpoint["model1_state_disct"])
        self._model2.load_state_dict(checkpoint["model2_state_disct"])
        self._optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self._optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
        else:
            self._model1.train()
            self._model2.train()
        return checkpoint["epoch"]