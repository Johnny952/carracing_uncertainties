import torch
from torch.distributions import Beta
import numpy as np

from uncert_model import Model
from utilities import Buffer

class Agent():
    """
    Agent for training
    """
    def __init__(
        self, nb_nets, img_stack, gamma, from_checkpoint=False,
        model="base", max_grad_norm=0.5, 
        clip_param=0.1, ppo_epoch=10, buffer_capacity=2000, 
        batch_size=128, lr=1e-3, device='cpu'):
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param  # epsilon in clipped loss
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity, self.batch_size = buffer_capacity, batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device

        self.training_step = 0
        self._model = Model(nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device=device, model=model)
        self.model_name="base"

        self._buffer = Buffer(img_stack, buffer_capacity, device=device)
    
    def select_action(self, state, eval=False):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)

        (alpha, beta), _, (epistemic, aleatoric) = self._model.forward_nograd(state)
        
        if eval:
            action = alpha / (alpha + beta)
            a_logp = 0

            action = action.squeeze().cpu().numpy()
        else:
            dist = Beta(alpha, beta)
            action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1)

            action = action.squeeze().cpu().numpy()
            a_logp = a_logp.item()
        return action, a_logp, (epistemic, aleatoric)

    def save_param(self, epoch):
        self._model.save_model(epoch=epoch)

    def load_param(self, eval_mode=False):
        return self._model.load_model(eval_mode=eval_mode)
    
    def store_transition(self, transition):
        return self._buffer.store(transition)
    
    def update(self):
        self.training_step += 1

        self._model.train(self.ppo_epoch, self.clip_param, self._buffer.sample())