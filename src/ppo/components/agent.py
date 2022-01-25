import torch
from torch.distributions import Beta
import numpy as np

from .uncert_model import make_model
from utilities import Buffer


class Agent:
    """
    Agent for training
    """

    def __init__(
        self,
        nb_nets,
        img_stack,
        gamma,
        model="base",
        max_grad_norm=0.5,
        clip_param=0.1,
        ppo_epoch=10,
        buffer_capacity=2000,
        batch_size=128,   # 128
        lr=1e-3,
        device="cpu",
    ):
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param  # epsilon in clipped loss
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity, self.batch_size = buffer_capacity, batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device

        self.training_step = 0
        self._model = make_model(
            nb_nets,
            lr,
            img_stack,
            gamma,
            batch_size,
            buffer_capacity,
            device=device,
            model=model,
        )
        self.model_name = "base"

        self._buffer = Buffer(img_stack, buffer_capacity, device=device)

    def select_action(self, state, eval=False):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)

        (alpha, beta), _, (epistemic, aleatoric) = self._model.get_uncert(state)

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

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        self._model.save(epoch=epoch, path=path)

    def load(self, path, eval_mode=False):
        return self._model.load(path, eval_mode=eval_mode)

    def store_transition(self, transition):
        return self._buffer.store(transition)

    def able_sample(self):
        return self._buffer.able_sample()

    def update(self):
        self.training_step += 1

        self._model.train(self.ppo_epoch, self.clip_param, self._buffer.sample())
