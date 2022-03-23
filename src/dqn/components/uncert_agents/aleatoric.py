import torch

from components.uncert_agents.abstact import AbstactAgent
from models.aleatoric import Aleatoric
from shared.utils.losses import det_loss

class AleatoricAgent(AbstactAgent):
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
        super(AleatoricAgent, self).__init__(
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

        self._weight_decay = 1e-4
        self._model1 = Aleatoric(img_stack, len(actions)).to(self._device)
        self._model2 = Aleatoric(img_stack, len(actions)).to(self._device)

        self._optimizer1 = torch.optim.Adam(self._model1.parameters(), lr=self._lr)
        self._optimizer2 = torch.optim.Adam(self._model2.parameters(), lr=self._lr)

        self._criterion = det_loss
    
    def get_values(self, observation):
        _, values, log_var = self._model1(observation)
        _, index = torch.max(values, dim=-1)
        epistemic = torch.Tensor([0])
        aleatoric = log_var[0, index[0].cpu().numpy()]
        return index, epistemic, aleatoric
    
    def compute_loss(self, states, actions, next_states, rewards, dones):
        reparametrization1, mu1, log_var1 = self._model1(states)
        reparametrization2, mu2, log_var2 = self._model2(states)
        curr_Q1 = mu1.gather(1, actions).squeeze(dim=-1)
        curr_Q2 = mu2.gather(1, actions).squeeze(dim=-1)

        curr_reparametrization1 = reparametrization1.gather(1, actions).squeeze(dim=-1)
        curr_reparametrization2 = reparametrization2.gather(1, actions).squeeze(dim=-1)
        curr_log_var1 = log_var1.gather(1, actions).squeeze(dim=-1)
        curr_log_var2 = log_var2.gather(1, actions).squeeze(dim=-1)

        next_Q = torch.min(
            torch.max(self._model1(next_states)[1], 1)[0],
            torch.max(self._model2(next_states)[1], 1)[0],
        ).squeeze(dim=-1)
        expected_Q = rewards + (1 - dones) * self._gamma * next_Q

        loss1 = self._criterion(curr_reparametrization1, expected_Q.detach(), curr_Q1, curr_log_var1, weight_decay=self._weight_decay)
        loss2 = self._criterion(curr_reparametrization2, expected_Q.detach(), curr_Q2, curr_log_var2, weight_decay=self._weight_decay)

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