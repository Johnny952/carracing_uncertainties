import copy
import torch
import torch.nn as nn
import numpy as np

from utils import ReplayMemory
from models import Net


class Agent:
    def __init__(self, img_stack, nb_actions, learning_rate, gamma, epsilon, nb_training_steps, buffer_capacity, batch_size, device='cpu', epsilon_decay=True):
        """Constructor of Agent class

        Args:
            img_stack (int): Number of last state frames to stack
            nb_actions (int): Number of discrete actions
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
        
        self._buffer = ReplayMemory(buffer_capacity, batch_size)

        self._lr = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._use_eps_decay = epsilon_decay

        self._img_stack = img_stack
        self._nb_actions = nb_actions

        # Discritize action space

        if epsilon_decay:
            self._epsilon_min = 0
            self._epsilon_decay = self._epsilon / (nb_training_steps / 2.)
        
        self._net = Net(img_stack, nb_actions).to(self._device)
        self._target_net = None
        self.replace_target_network()
        
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def store_transition(self, state, action, next_state, reward, done):
        """Store a transition in replay buffer

        Args:
            state (np.ndarray): State in time t
            action (int): Action index taken in time t
            next_state (np.ndarray): State in time t+1
            reward (float): Reward of moving from state t to t+1
            done (bool): Whether state in time t+1 is terminal or not
        """        
        self._buffer.push(state, action, next_state, reward, done)

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
                values = self._net(torch.from_numpy(observation).float().to(self._device))
                _, action = torch.max(values, dim=0)
        else:
            # Select random action
            action = torch.randint(0, int(self._nb_actions), size=(1,))[0]

        if not greedy and self._epsilon >= self._epsilon_min:
            # Implement epsilon linear decay
            self._epsilon -= self._epsilon_decay
        return action.cpu().numpy()
    
    def update(self):
        """Trains agent model
        """        
        for _ in range(1):
            buffer_samples = self._buffer.sample()
            print(buffer_samples)


if __name__ == "__main__":
    import gym
    from utils import imgstackRGB2graystack

    env = gym.make('CarRacing-v0')
    img_stack = 4
    env = gym.wrappers.FrameStack(env, img_stack)
    
    agent = Agent(img_stack, 10, 0.001, 0.1, 0.5, 1000, 200, 32)

    state = imgstackRGB2graystack(env.reset())

    for _ in range(300):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = imgstackRGB2graystack(next_state)
        agent.store_transition(state, action, next_state, reward, done)
        state = next_state
        if done:
            break

    agent.update