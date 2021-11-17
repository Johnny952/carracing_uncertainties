import numpy as np

def add_noise(state, dev):
    noise = np.random.normal(loc=0, scale=dev, size=state.shape)
    noisy_state = state + noise
    noisy_state[noisy_state > 1] = 1
    noisy_state[noisy_state < -1] = -1
    return noisy_state

def add_random_std_noise(state, upper, lower):
    std = np.random.uniform(lower, upper)
    return add_noise(state, std)