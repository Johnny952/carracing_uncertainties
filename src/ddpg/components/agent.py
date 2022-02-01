import gc
import logging
import os
import torch
from collections import namedtuple
import numpy as np

import torch.nn.functional as F
from torch.optim import Adam

from models.actor import Actor
from models.critic import Critic
from shared.utils.replay_buffer import ReplayMemory

logger = logging.getLogger('ddpg')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(object):

    def __init__(
        self,
        gamma,
        tau,
        img_stack,
        buffer_capacity,
        batch_size,
        action_space=[-1, 1],
        device='cpu',
        actor_lr=1e-4,
        critic_lr=1e-3,
        critic_weight_decay=1e-2
    ):
        """
        Deep Deterministic Policy Gradient
        Read the detail about it here:
        https://arxiv.org/abs/1509.02971
        Arguments:
            gamma:          Discount factor
            tau:            Update factor for the actor and the critic
            img_stack:      Number of states stacked.
            action_space:   The action space of the used environment. Used to clip the actions and 
                            to distinguish the number of outputs
        """
        self.device = device

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # Define buffer capacity
        self._Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )
        self._buffer = ReplayMemory(buffer_capacity, batch_size, self._Transition)

        # Define the actor
        self.actor = Actor(img_stack, output_dim=self.action_space).to(device)
        self.actor_target = Actor(img_stack, output_dim=self.action_space).to(device)

        # Define the critic
        self.critic = Critic(img_stack).to(device)
        self.critic_target = Critic(img_stack).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=actor_lr)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=critic_lr,
                                     weight_decay=critic_weight_decay
                                     )  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self._nb_update = 0

    def select_action(self, state, action_noise=None):
        """
        Evaluates the action to perform in a given state
        Arguments:
            state:          State to perform the action on in the env. 
                            Used to evaluate the action.
            action_noise:   If not None, the noise to apply on the evaluated action
        """
        x = state.to(self.device)

        # Get the continous action value to perform in the env
        # self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        # self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(self.device)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space[0], self.action_space[0])

        return mu

    def update(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update
        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        # Get tensors from the batch
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.unpack(batch)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def unpack(self, batch):
        states = torch.cat(batch.state).float().to(self._device)
        actions = torch.cat(batch.action).long().to(self._device)
        next_states = torch.cat(batch.next_state).float().to(self._device)
        rewards = torch.cat(batch.reward).to(self._device)
        dones = torch.cat(batch.done).to(self._device)

        return states, actions, next_states, rewards, dones

    def save_checkpoint(self, last_timestep, replay_buffer, path="param/ppo_net_param.pkl"):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'
        Arguments:
            last_timestep:  Last timestep in training before saving
            replay_buffer:  Current replay buffer
        """
        logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': replay_buffer,
        }
        logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, path)
        gc.collect()
        logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def load_checkpoint(self, checkpoint_path):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.
        Arguments:
            checkpoint_path:    File to load the model from
        """

        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint...({})".format(checkpoint_path))

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        """
        Sets the model in evaluation mode
        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode
        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def store_transition(self, state, action, next_state, reward, done):
        """Store a transition in replay buffer

        Args:
            state (np.ndarray): State in time t
            action (torch.Tensot): Action index taken in time t
            next_state (np.ndarray): State in time t+1
            reward (float): Reward of moving from state t to t+1
            done (bool): Whether state in time t+1 is terminal or not
        """
        self._buffer.push(
            torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(dim=0),
            action.unsqueeze(dim=0),
            torch.from_numpy(np.array(next_state, dtype=np.float32)).unsqueeze(dim=0),
            torch.Tensor([reward]),
            torch.Tensor([done]),
        )
        return self._buffer.able_sample()