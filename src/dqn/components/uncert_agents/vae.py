import torch
import torch.nn as nn
import wandb

from components.uncert_agents.abstact import AbstactAgent
from models.base import Net
from ppo.models.vae import VanillaVAE

class VaeAgent(AbstactAgent):
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
        super(VaeAgent, self).__init__(
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

        self._criterion = nn.MSELoss()

        latent_size = 128
        self._nb_vae_update = 0
        self._kld_scale = 0.015
        self._vae = VanillaVAE(latent_size).float().to(self._device)
        self._vae_optimizer = torch.optim.Adam(self._vae.parameters(), lr=0.001)

        self._model1 = Net(img_stack, len(actions)).to(self._device)
        self._model2 = Net(img_stack, len(actions)).to(self._device)

        self._optimizer1 = torch.optim.Adam(self._model1.parameters(), lr=self._lr)
        self._optimizer2 = torch.optim.Adam(self._model2.parameters(), lr=self._lr)
    
    def get_values(self, observation):
        values = self._model1(observation)
        _, index = torch.max(values, dim=-1)
        [_, log_var] = self._vae.encode(observation)
        epistemic = torch.sum(torch.exp(log_var))
        aleatoric = torch.Tensor([0])
        return index, epistemic, aleatoric

    def update(self):
        states = super().update()[0]
        
        vae_loss, recons_loss, kld_loss = self.vae_update(states)

        self.log_vae_loss(vae_loss, recons_loss, kld_loss)
        self._nb_vae_update += 1

    def log_vae_loss(self, vae_loss, recons_loss, kld_loss):
        wandb.log(
            {
                "VAE Update Step": self._nb_vae_update,
                "VAE Loss": vae_loss,
                "Reconst Loss": recons_loss,
                "KLD Loss": kld_loss,
            }
        )

    def vae_update(self, states):
        [decoding, input, mu, log_var] = self._vae(states)
        l = self._vae.loss_function(decoding, input, mu, log_var, M_N=self._kld_scale)
        loss = l['loss']
        recons_loss = l['Reconstruction_Loss']
        kld_loss = l['kld_loss']
        self._vae_optimizer.zero_grad()
        loss.backward()
        self._vae_optimizer.step()
        return loss.item(), recons_loss, kld_loss

    
    def save_param(self, epoch, path="param/ppo_net_param.pkl"):
        tosave = {
            "epoch": epoch,
            "model1_state_disct": self._model1.state_dict(),
            "model2_state_disct": self._model2.state_dict(),
            "optimizer1_state_dict": self._optimizer1.state_dict(),
            "optimizer2_state_dict": self._optimizer2.state_dict(),
            'vae_state_dict': self._vae.state_dict(),
            'vae_optimizer_state_dict': self._vae_optimizer.state_dict(),
        }
        torch.save(tosave, path)

    def load_param(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model1.load_state_dict(checkpoint["model1_state_disct"])
        self._model2.load_state_dict(checkpoint["model2_state_disct"])
        self._optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self._optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])
        self._vae.load_state_dict(checkpoint['vae_state_dict'])
        self._vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
            self._vae.eval()
        else:
            self._model1.train()
            self._model2.train()
            self._vae.train()
        return checkpoint["epoch"]