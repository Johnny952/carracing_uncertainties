import copy
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple

from utils import ReplayMemory
from models import Net


class Agent:
    def __init__(self, img_stack, actions, learning_rate, gamma, epsilon, buffer_capacity, batch_size, device='cpu', epsilon_decay=0.999, clip_grad=False, epsilon_min=0):
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
            epsilon_decay (float, optional): Epsilon decay factor. Defaults to 0.999.
        """        
        self._device = device
        self._clip_grad = clip_grad
        
        self._Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
        self._buffer = ReplayMemory(buffer_capacity, batch_size, self._Transition)

        self._lr = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

        self._img_stack = img_stack
        self._actions = actions

        self._net = Net(img_stack, len(actions)).to(self._device)
        self._target_net = None
        self.replace_target_network()
        
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)
        self._criterion = nn.MSELoss()

        self.K = 0.05
    
    def empty_buffer(self):
        """Empty replay buffer"""        
        self._buffer.empty()
    def nuber_experiences(self):
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
            torch.from_numpy(np.array(state)).unsqueeze(dim=0), 
            action.unsqueeze(dim=0), 
            torch.from_numpy(np.array(next_state)).unsqueeze(dim=0), 
            torch.Tensor([reward]), 
            torch.Tensor([done])
            )

    def replace_target_network(self):
        """Replace target network with actual network
        """        
        # self._target_net.load_state_dict(self._net.state_dict())
        self._target_net = copy.deepcopy(self._net).to(self._device)
        # self._target_deepq_network.eval()
        for p in self._target_net.parameters():
            p.requires_grad = False
        
    def select_action(self, observation, greedy=False):
        """Selects a epislon greedy action

        Args:
            observation (np.ndarray): Observation of the environment
            greedy (bool, optional): Whether to use only greedy actions or not. Defaults to False.

        Returns:
            int: The action taken
            int: The corresponding action index
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
        return action, index.cpu()
    
    def epsilon_step(self):
        """Epsilon decay e = e*factor"""        
        if self._epsilon >= self._epsilon_min:
            # epsilon  decay
            self._epsilon = (self._epsilon - self._epsilon_min) * self._epsilon_decay + self._epsilon_min
            #self._epsilon *= self._epsilon_decay
    
    def update(self):
        """Trains agent model"""
        for _ in range(1):
            transitions = self._buffer.sample()
            batch = self._Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).float().to(self._device)
            action_batch = torch.cat(batch.action).long().to(self._device)
            next_state_batch = torch.cat(batch.next_state).float().to(self._device)
            reward_batch = torch.cat(batch.reward).to(self._device)
            done_batch = torch.cat(batch.done).to(self._device)

            state_action_values = self._net(state_batch).gather(1, action_batch).squeeze(dim=-1)
            next_state_values = ((1 - done_batch)*self._target_net(next_state_batch).max(1)[0]).detach()
            expected_state_action_values = (next_state_values * self._gamma) + reward_batch

            loss = self._criterion(state_action_values, expected_state_action_values) + self.K*torch.mean(state_action_values)
            self._optimizer.zero_grad()
            loss.backward()
            if self._clip_grad:
                for param in self._net.parameters():
                    param.grad.data.clamp_(-1, 1)
            self._optimizer.step()
    
    def save_param(self, epoch):
        """Save agent's parameters

        Args:
            epoch (int): Training epoch
        """        
        tosave = {
            'epoch': epoch,
            'model_state_disct': self._net.state_dict(),
            'target_model_state_dict': self._target_net.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }
        torch.save(tosave, 'param/ppo_net_param.pkl')
    
    def load_param(self, path, eval_mode=False):
        """Load Agent checkpoint

        Args:
            path (str): Path to file
            eval_mode (bool, optional): Whether to use agent in eval mode or not. Defaults to False.

        Returns:
            int: checkpoint epoch
        """        
        checkpoint = torch.load(path)
        self._net.load_state_dict(checkpoint['model_state_disct'])
        self._target_net.load_state_dict(checkpoint['target_model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if eval_mode:
            self._net.eval()
        else:
            self._net.train()
        return checkpoint['epoch']
    
    def eval_mode(self):
        """Prediction network in evaluation mode"""        
        self.net._eval()
    def train_mode(self):
        """Prediction network in training mode"""   
        self._net.train()


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