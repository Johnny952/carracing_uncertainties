import numpy as np
import torch

class Buffer:
    def __init__(self, img_stack, buffer_capacity, device='cpu'):
        self.transition = np.dtype([('s', np.float64, (img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                                    ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96))])
        self.buffer = np.empty(buffer_capacity, dtype=self.transition)
        self.counter = 0
        self.buffer_capacity = buffer_capacity
        self.device = device

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False
    
    def sample(self):
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        return s, a, r, s_, old_a_logp