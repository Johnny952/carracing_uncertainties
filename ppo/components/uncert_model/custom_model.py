from torch import optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
import torch.nn as nn
import torch
from .basic_model import BaseTrainerModel
from utilities import flow_loss_split
from models import FlowModel, FCNDecoder, FCNEncoder
import wandb
import sys
sys.path.append('..')


class CustomTrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(CustomTrainerModel, self).__init__(nb_nets, lr,
                                                 img_stack, gamma, batch_size, buffer_capacity, device=device)

        latent_size = 64
        self._encoder = FCNEncoder(
            img_stack, output_dim=latent_size).double().to(self.device)
        self._flow_model = FlowModel(
            flows=['PlanarFlow'] * 10, D=latent_size, device=self.device).double().to(self.device)
        self._decoder = FCNDecoder(
            img_stack, input_dim=latent_size).double().to(self.device)

        parameters = list(self._encoder.parameters()) \
            + list(self._flow_model.parameters())\
            + list(self._decoder.parameters())
        self._flow_optimizer = optim.Adam(parameters, lr=lr)

        self._loss_autoencoding = nn.MSELoss()
        self._nb_flow_epochs = 10
        self._nb_flow_update = 0
        self._encode_loss_scale = 1

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

        for _ in range(self._nb_flow_epochs):
            sampler = SubsetRandomSampler(range(self.buffer_capacity))
            flow_loss, encode_loss, loss = self.train_once_flow(s, a, sampler)
            self.log_flow_loss(flow_loss, encode_loss, loss)
            self._nb_flow_update += 1

    def log_flow_loss(self, flow_loss, encode_loss, loss):
        to_log = {
            'Update Flow Step': self._nb_flow_update,
            'Flow Loss': float(flow_loss),
            'Encode Loss': float(encode_loss),
            'Total Norm Flow Loss': float(loss),
        }
        wandb.log(to_log)

    def train_once_flow(self, s, a, rand_sampler):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)
        acc_flow_loss = 0
        acc_encode_loss = 0
        acc_loss = 0
        for index in sampler:
            # Normalizing flow training
            mu, log_sigma = self._encoder(s[index])
            z_k, log_prob_z0, log_prob_zk, log_det = self._flow_model(
                mu, log_sigma)
            x_hat = self._decoder(z_k)
            norm_flow_loss, encode_loss = flow_loss_split(log_prob_z0, log_prob_zk, log_det,
                                                          x_hat, s[index], self._loss_autoencoding)
            loss = norm_flow_loss + self._encode_loss_scale * encode_loss
            self._flow_optimizer.zero_grad()
            loss.backward()
            self._flow_optimizer.step()

            acc_flow_loss += norm_flow_loss.item()
            acc_encode_loss += encode_loss.item()
            acc_loss += loss.item()

        return acc_flow_loss, acc_encode_loss, acc_loss

    def save(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'encoder': self._encoder.state_dict(),
            'flow': self._flow_model.state_dict(),
            'decoder': self._decoder.state_dict(),
            'flow_optimizer': self._flow_optimizer.state_dict()
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._encoder.load_state_dict(checkpoint['encoder'])
        self._flow_model.load_state_dict(checkpoint['flow'])
        self._decoder.load_state_dict(checkpoint['decoder'])
        self._flow_optimizer.load_state_dict(checkpoint['flow_optimizer'])
        if eval_mode:
            self._model.eval()
            self._encoder.eval()
            self._flow_model.eval()
            self._decoder.eval()
        else:
            self._model.train()
            self._encoder.train()
            self._flow_model.train()
            self._decoder.train()
        return checkpoint['epoch']

    def get_uncert(self, state):
        with torch.no_grad():
            mu, log_sigma = self._encoder(state)
            z_k, log_prob_z0, log_prob_zk, log_det = self._flow_model(
                mu, log_sigma)
            # x_hat = self._decoder(z_k)

        # TODO: Aleatoric estimation
        epistemic = 1 - torch.exp(log_prob_z0)
        aleatoric = torch.Tensor([0])

        (alpha, beta), v = self.forward_nograd(state)
        return (alpha, beta), v, (epistemic, aleatoric)
