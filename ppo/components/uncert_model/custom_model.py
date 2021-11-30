from torch.utils.data.sampler import BatchSampler
from torch.distributions import Beta
import torch.nn as nn
import torch
from .basic_model import BaseTrainerModel
from utilities import flow_loss
from models import FlowModel, FCNDecoder, FCNEncoder
import sys
sys.path.append('..')


class CustomTrainerModel(BaseTrainerModel):
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu'):
        super(CustomTrainerModel, self).__init__(nb_nets, lr,
                                                    img_stack, gamma, batch_size, buffer_capacity, device=device)
        
        D = 40
        self._encoder = FCNEncoder(hidden_sizes=[128, 64, 2*D], dim_input=28*28)
        self._flow_model = FlowModel(flows=['PlanarFlow'] * 10, D=40)
        self._decoder = FCNDecoder(hidden_sizes=[64, 128, 784], dim_input=40)
        # TODO: Crear modelo que tiene encoder, flow model y decoder, acá dejar el modelo que los contiene con su optimizador

        self._loss_fn = flow_loss
        self._loss_autoencoding = nn.MSELoss
        self._value_scale = 1

    def train_once(self, net, optimizer, target_v, adv, old_a_logp, s, a, clip_param, rand_sampler):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)
        acc_action_loss = 0
        acc_value_loss = 0
        acc_loss = 0

        for index in sampler:
            # Same base model training
            prediction = net(s[index])
            alpha, beta = prediction[0]
            dist = Beta(alpha, beta)
            a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
            ratio = torch.exp(a_logp - old_a_logp[index])

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                1.0 + clip_param) * adv[index]
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.get_value_loss(prediction, target_v[index])
            loss = action_loss + 2. * value_loss

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
            optimizer.step()


            # Normalizing flow training
            # TODO: training


            acc_action_loss += action_loss.item()
            acc_value_loss += value_loss.item()
            acc_loss += loss.item()
        return acc_loss, acc_action_loss, acc_value_loss

    def save(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'encoder': self._encoder.state_dict(),
            'flow': self._flow_model.state_dict(),
            'decoder': self._decoder.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._encoder.load_state_dict(checkpoint['encoder'])
        self._flow_model.load_state_dict(checkpoint['flow'])
        self._decoder.load_state_dict(checkpoint['decoder'])
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
            z_k, log_prob_z0, log_prob_zk, log_det = self._flow_model(mu, log_sigma)
            # x_hat = self._decoder(z_k)

        # TODO: Aleatoric estimation
        epistemic = torch.exp(log_prob_z0)
        aleatoric = torch.Tensor([0])

        (alpha, beta), v = self.forward_nograd(state)
        return (alpha, beta), v, (epistemic, aleatoric)
