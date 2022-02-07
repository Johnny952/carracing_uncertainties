import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import wandb

from models.base import Net
from utilities.eps_scheduler import Epsilon
from utilities.replay_buffer import ReplayMemory


class Agent:
    def __init__(
        self,
        img_stack,
        actions,
        learning_rate,
        gamma,
        buffer_capacity,
        batch_size,
        device="cpu",
        clip_grad=False,
        epsilon_method="linear",
        epsilon_max=1,
        epsilon_min=0.1,
        epsilon_factor=3,
        epsilon_max_steps=1000,
    ):

        self._device = device
        self._clip_grad = clip_grad
        self._batch_size = batch_size
        self._lr = learning_rate
        self._gamma = gamma
        self._img_stack = img_stack
        self._actions = actions
        self._criterion = nn.MSELoss()
        self._epsilon = Epsilon(
            epsilon_max_steps,
            method=epsilon_method,
            epsilon_max=epsilon_max,
            epsilon_min=epsilon_min,
            factor=epsilon_factor,
        )

        self._Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
        )
        self._buffer = ReplayMemory(buffer_capacity, batch_size, self._Transition)

        self._model1 = Net(img_stack, len(actions)).to(self._device)
        self._model2 = Net(img_stack, len(actions)).to(self._device)

        self._optimizer1 = torch.optim.Adam(self._model1.parameters(), lr=self._lr)
        self._optimizer2 = torch.optim.Adam(self._model2.parameters(), lr=self._lr)

        self._nb_update = 0

    def empty_buffer(self):
        """Empty replay buffer"""
        self._buffer.empty()

    def number_experiences(self):
        """Get number of saved experiences

        Returns:
            int: Number of experiences
        """
        return len(self._buffer)

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

    def epsilon_step(self):
        """Epsilon decay e = e*factor"""
        self._epsilon.step()

    def epsilon(self):
        return self._epsilon.epsilon()

    def select_action(self, observation, greedy=False):
        """Selects a epislon greedy action

        Args:
            observation (np.ndarray): Observation of the environment
            greedy (bool, optional): Whether to use only greedy actions or not. Defaults to False.

        Returns:
            int: The action taken
            int: The corresponding action index
        """
        if greedy or np.random.rand() > self._epsilon.epsilon():
            # Select action greedily
            self._model1.batchnorm_state(activate=False)
            with torch.no_grad():
                values = self._model1(
                    (torch.from_numpy(observation).unsqueeze(dim=0).float()).to(
                        self._device
                    )
                )
                _, index = torch.max(values, dim=-1)
            self._model1.batchnorm_state(activate=True)
        else:
            # Select random action
            index = torch.randint(0, len(self._actions), size=(1,))
        return self._actions[index], index.cpu()

    def compute_loss(self, states, actions, next_states, rewards, dones):
        curr_Q1 = self._model1(states).gather(1, actions).squeeze(dim=-1)
        curr_Q2 = self._model2(states).gather(1, actions).squeeze(dim=-1)

        # next_Q1 = self._model1(next_states)
        # next_Q2 = self._model2(next_states)
        next_Q = torch.min(
            torch.max(self._model1(next_states), 1)[0],
            torch.max(self._model2(next_states), 1)[0],
        ).squeeze(dim=-1)
        expected_Q = rewards + (1 - dones) * self._gamma * next_Q

        loss1 = self._criterion(curr_Q1, expected_Q.detach())
        loss2 = self._criterion(curr_Q2, expected_Q.detach())

        return loss1, loss2

    def update(self):
        states, actions, next_states, rewards, dones = self.unpack(
            self._buffer.sample()
        )
        loss1, loss2 = self.compute_loss(states, actions, next_states, rewards, dones)

        self._optimizer1.zero_grad()
        loss1.backward()
        if self._clip_grad:
            for param in self._model1.parameters():
                param.grad.data.clamp_(-1, 1)
        self._optimizer1.step()

        self._optimizer2.zero_grad()
        loss2.backward()
        if self._clip_grad:
            for param in self._model2.parameters():
                param.grad.data.clamp_(-1, 1)
        self._optimizer2.step()

        # self.epsilon_step()
        self.log_loss(loss1.item(), loss2.item())
        self._nb_update += 1

    def log_loss(self, loss1, loss2):
        wandb.log(
            {
                "Update Step": self._nb_update,
                "Loss 1": float(loss1),
                "Loss 2": float(loss2),
                "Epsilon": float(self.epsilon()),
            }
        )

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
            self._model.eval()
        else:
            self._model.train()
        return checkpoint["epoch"]

    def eval_mode(self):
        self._model1._eval()
        self._model2._eval()

    def train_mode(self):
        self._model1.train()
        self._model2.train()

    def unpack(self, batch):
        states = torch.cat(batch.state).float().to(self._device)
        actions = torch.cat(batch.action).long().to(self._device)
        next_states = torch.cat(batch.next_state).float().to(self._device)
        rewards = torch.cat(batch.reward).to(self._device)
        dones = torch.cat(batch.done).to(self._device)

        return states, actions, next_states, rewards, dones

if __name__ == "__main__":
    import gym
    from utilities import imgstackRGB2graystack

    env = gym.make("CarRacing-v0")
    img_stack = 4
    env = gym.wrappers.FrameStack(env, img_stack)

    low_state = env.action_space.low
    high_state = env.action_space.high

    posible_actions = (
        [-1, 0, 0],  # Turn Left
        [1, 0, 0],  # Turn Right
        [0, 0, 1],  # Full Break
        [0, 1, 0],  # Accelerate
        [0, 0, 0],  # Do nothing
    )

    agent = Agent(img_stack, posible_actions, 0.001, 0.1, 0.5, 1000, 200, 32)

    state = imgstackRGB2graystack(env.reset())

    for _ in range(300):
        action, action_index = agent.select_action(state)
        next_state, reward, done = env.step(action)[:3]
        next_state = imgstackRGB2graystack(next_state)
        agent.store_transition(state, action_index, next_state, reward, done)
        state = next_state
        if done:
            break

    agent.update()
