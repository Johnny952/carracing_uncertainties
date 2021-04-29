from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity, batch_size):
        """Constructor of Replay Buffer

        Args:
            capacity (int): Maximum number of experiences
            batch_size (int): Number of experiences to sample
        """        
        self.memory = deque([],maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        """Save a experiences"""
        self.memory.append(Transition(*args))

    def sample(self):
        """Sample experiences

        Raises:
            Exception: Number of experiences is less than the required

        Returns:
            list: Sample of experiences (state, action, next state, reward, done)
        """        
        if len(self) < self.batch_size:
            raise Exception('Number of experiences is less than the required')
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    import numpy as np

    buffer = ReplayMemory(50, 16)

    for i in range(60):
        state = np.random.randn(96, 96, 4)
        action = 3
        next_state = np.random.randn(96, 96, 4)
        reward = -1
        done = False

        buffer.push(state, action, next_state, reward, done)

        print(f"Experiences: {i+1}\tSaved: {len(buffer)}")
    
    print("\nSample:", len(buffer.sample()))
