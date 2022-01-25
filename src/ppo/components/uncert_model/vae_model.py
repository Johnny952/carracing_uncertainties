from torch import optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
from components.uncert_model.basic_model import BaseTrainerModel
from models.vae import VanillaVAE
import wandb


class VAETrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(VAETrainerModel, self).__init__(nb_nets, lr,
                                                 img_stack, gamma, batch_size, buffer_capacity, device=device)

        latent_size = 128
        self._vae = VanillaVAE(latent_size).double().to(self.device)

        self._vae_optimizer = optim.Adam(self._vae.parameters(), lr=lr)

        self._nb_vae_epochs = 10
        self._nb_vae_update = 0
        self._kld_scale = 0.005

    def train(self, epochs, clip_param, database):
        (s, a, r, s_, old_a_logp) = database
        target_v = r + self.gamma * self.forward_nograd(s_)[1]
        adv = target_v - self.forward_nograd(s)[1]

        for _ in range(epochs):
            sampler = SubsetRandomSampler(range(self.buffer_capacity))
            loss, action_loss, value_loss = self.train_once(
                self._model, self._optimizer, target_v, adv, old_a_logp, s, a, clip_param, sampler)
            self.log_loss(loss, action_loss, value_loss)
            self._nb_update += 1

        for _ in range(self._nb_vae_epochs):

            sampler = SubsetRandomSampler(range(self.buffer_capacity))
            vae_loss, encode_loss, loss = self.train_once_vae(s, a, sampler)
            self.log_vae_loss(vae_loss, encode_loss, loss)
            self._nb_vae_update += 1

    def log_vae_loss(self, vae_loss, encode_loss, loss):
        to_log = {
            'Update VAE Step': self._nb_vae_update,
            'KLD Loss': float(vae_loss),
            'Reconst Loss': float(encode_loss),
            'Total VAE Loss': float(loss),
        }
        wandb.log(to_log)

    def train_once_vae(self, s, a, rand_sampler):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)
        acc_vae_loss = 0
        acc_encode_loss = 0
        acc_loss = 0
        for index in sampler:
            # Normalizing flow training
            [decoding, input, mu, log_var] = self._vae(s[index])
            l = self._vae.loss_function(decoding, input, mu, log_var, M_N=self._kld_scale)
            loss = l['loss']
            recons_loss = l['Reconstruction_Loss']
            kld_loss = l['kld_loss']
            self._vae_optimizer.zero_grad()
            loss.backward()
            self._vae_optimizer.step()

            acc_vae_loss += kld_loss.item()
            acc_encode_loss += recons_loss.item()
            acc_loss += loss.item()

        return acc_vae_loss, acc_encode_loss, acc_loss

    def save(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'vae_state_dict': self._vae.state_dict(),
            'vae_optimizer_state_dict': self._vae_optimizer.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._vae.load_state_dict(checkpoint['vae_state_dict'])
        self._vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        if eval_mode:
            self._model.eval()
            self._vae.eval()
        else:
            self._model.train()
            self._vae.train()
        return checkpoint['epoch']

    def get_uncert(self, state):
        with torch.no_grad():
            [_, log_var] = self._vae.encode(state)

        # TODO: Aleatoric estimation
        epistemic = -torch.sum(log_var)
        aleatoric = torch.Tensor([0])

        (alpha, beta), v = self.forward_nograd(state)
        return (alpha, beta), v, (epistemic, aleatoric)
