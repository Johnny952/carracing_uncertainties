import torch
import torch.nn as nn
import wandb

from components.uncert_agents.abstact import AbstactAgent
from models.bnn import BayesianModel

class BnnAgent(AbstactAgent):
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
        super(BnnAgent, self).__init__(
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
        self.sample_nbr = 50
        self.complexity_cost_weight = 100

        self._model1 = BayesianModel(img_stack, len(actions)).to(self._device)
        self._model2 = BayesianModel(img_stack, len(actions)).to(self._device)

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
        epistemic = torch.var(values_list, dim=0)
        aleatoric = torch.Tensor([0])
        return index, epistemic, aleatoric
    
    def compute_loss(self, states, actions, next_states, rewards, dones):
        loss1 = torch.tensor([0.], dtype=torch.float, requires_grad=True).to(self._device)
        loss2 = torch.tensor([0.], dtype=torch.float, requires_grad=True).to(self._device)

        for _ in range(self.sample_nbr):
            curr_Q1 = self._model1(states).gather(1, actions).squeeze(dim=-1)
            curr_Q2 = self._model2(states).gather(1, actions).squeeze(dim=-1)

            next_Q = torch.min(
                torch.max(self._model1(next_states), 1)[0],
                torch.max(self._model2(next_states), 1)[0],
            ).squeeze(dim=-1)
            expected_Q = rewards + (1 - dones) * self._gamma * next_Q

            value_loss1 = self._criterion(curr_Q1, expected_Q.detach())
            value_loss2 = self._criterion(curr_Q2, expected_Q.detach())

            kld_loss1 = self._model1.nn_kl_divergence() * self.complexity_cost_weight
            kld_loss2 = self._model2.nn_kl_divergence() * self.complexity_cost_weight
            loss1 += kld_loss1 + value_loss1
            loss2 += kld_loss2 + value_loss2

            wandb.log(
                {
                    "Value Loss 1": float(value_loss1),
                    "Value Loss 2": float(value_loss2),
                    "KLD Loss 2": float(kld_loss1),
                    "KLD Loss 2": float(kld_loss2),
                }
            )

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