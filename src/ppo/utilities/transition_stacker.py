from collections import namedtuple, deque
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.States = namedtuple('States', ('state'))
    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.States(*args))
    def __len__(self):
        return len(self.memory)
    def sample(self):
        return np.stack(self.memory.sample()).squeeze()