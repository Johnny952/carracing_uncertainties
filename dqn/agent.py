import copy
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple

from utils import ReplayMemory
from models import Net


class Agent:
    def __init__(self, img_stack, actions, learning_rate, gamma, epsilon, nb_training_steps, buffer_capacity, batch_size, device='cpu', epsilon_decay=True):
        """Constructor of Agent class

        Args:
            img_stack (int): Number of last state frames to stack
            actions (tuple): Posible actions
            learning_rate (float): Learning rate
            gamma (float): Discount Factor
            epsilon (float): Epsilon Greedy parameter
            nb_training_steps (int): Number of training steps
            buffer_capacity (int): Buffer capacity
            batch_size (int): Batch size
            device (str, optional): Use cuda or cpu. Defaults to 'cpu'.
            epsilon_decay (bool, optional): Whether to use epsilon decay or not. Defaults to True.
        """        
        self._device = device
        
        self._Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
        self._buffer = ReplayMemory(buffer_capacity, batch_size, self._Transition)

        self._lr = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._use_eps_decay = epsilon_decay

        self._img_stack = img_stack
        self._actions = actions

        if epsilon_decay:
            self._epsilon_min = 0
            self._epsilon_decay = self._epsilon / (nb_training_steps / 2.)
        
        self._net = Net(img_stack, len(actions)).to(self._device)
        self._target_net = None
        self.replace_target_network()
        
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)
        self._criterion = nn.MSELoss()

    def store_transition(self, state, action, next_state, reward, done):
        """Store a transition in replay buffer

        Args:
            state (np.ndarray): State in time t
            action (int): Action index taken in time t
            next_state (np.ndarray): State in time t+1
            reward (float): Reward of moving from state t to t+1
            done (bool): Whether state in time t+1 is terminal or not
        """
        self._buffer.push(
            torch.from_numpy(state).unsqueeze(dim=0), 
            action.unsqueeze(dim=0), 
            torch.from_numpy(next_state).unsqueeze(dim=0), 
            torch.Tensor([reward]), 
            torch.Tensor([done])
            )

    def replace_target_network(self):
        """Replace target network with actual network
        """        
        self._target_net = copy.deepcopy(self._net).to(self._device)
        # self._target_deepq_network.eval()
        for p in self._target_net.parameters():
            p.requires_grad = False
        
    def select_action(self, observation, greedy=False):
        """[summary]

        Args:
            observation (np.ndarray): Observation of the environment
            greedy (bool, optional): Whether to use only greedy actions or not. Defaults to False.

        Returns:
            int: The action taken
        """        
        if np.random.random() > self._epsilon or greedy:
            # Select action greedily
            with torch.no_grad():
                values = self._net(torch.from_numpy(observation).unsqueeze(dim=0).float().to(self._device))
                _, index = torch.max(values, dim=-1)
        else:
            # Select random action
            index = torch.randint(0, len(self._actions), size=(1,))
        action = self._actions[index]
        if not greedy and self._epsilon >= self._epsilon_min:
            # Implement epsilon linear decay
            self._epsilon -= self._epsilon_decay
        return action, index
    
    def update(self):
        """Trains agent model
        """
        for _ in range(1):
            transitions = self._buffer.sample()
            batch = self._Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).float()
            action_batch = torch.cat(batch.action).long()
            next_state_batch = torch.cat(batch.next_state).float()
            reward_batch = torch.cat(batch.reward)
            done_batch = torch.cat(batch.done)

            state_action_values = self._net(state_batch).gather(1, action_batch)
            next_state_values = ((1 - done_batch)*self._target_net(next_state_batch).max(1)[0]).detach()
            expected_state_action_values = (next_state_values * self._gamma) + reward_batch

            loss = self._criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()


if __name__ == "__main__":
    import gym
    from utils import imgstackRGB2graystack

    env = gym.make('CarRacing-v0')
    img_stack = 4
    env = gym.wrappers.FrameStack(env, img_stack)

    low_state = env.action_space.low
    high_state = env.action_space.high

    posible_actions = (
        [-1, 0, 0],              # Turn Left
        [1, 0, 0],               # Turn Right
        [0, 0, 1],              # Full Break
        [0, 1, 0],              # Accelerate
        [0, 0, 0],              # Do nothing
        )

    agent = Agent(img_stack, posible_actions, 0.001, 0.1, 0.5, 1000, 200, 32)

    state = imgstackRGB2graystack(env.reset())

    for _ in range(300):
        action, action_index = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = imgstackRGB2graystack(next_state)
        agent.store_transition(state, action_index, next_state, reward, done)
        state = next_state
        if done:
            break

    agent.update()