import numpy as np
from math import sqrt

# TODO: Simple linear or exp standar deviation decay of white noise
class BaseNoise:
    def __init__(self, action_dimension, max_steps, method='linear', max_=1.0, min_=0.1, factor=3) -> None:
        method = method.lower()
        if method == "constant":
            self._get_noise = self.constant
        elif method == "linear":
            self._get_noise = self.linear
        elif method == "exp":
            self._get_noise = self.exp
        elif method == "inverse_sigmoid":
            self._get_noise = self.inverse_sigmoid
        else:
            raise NotImplementedError(
                "method must be constant, linear, exp, or inverse_sigmoid"
            )

        self.action_dimension = action_dimension
        self._max_steps = max_steps
        self._max = max_
        self._min = min_
        self._factor = factor

        self.reset()

    def reset(self):
        self._step = 0

    @property
    def std(self):
        self._step += 1
        return self._get_noise(self._step)
    
    def constant(self, step):
        if step >= self._max_steps:
            return self._min
        return self._max

    def linear(self, step):
        return max(
            self._min,
            self._max
            - (self._max - self._min) * step / self._max_steps,
        )

    def exp(self, step):
        return self._min + (self._max - self._min) * np.exp(
            -self._factor * step / self._max_steps
        )

    def inverse_sigmoid(self, step):
        return self._min + (self._max - self._min) * (1 - 1 / (1 + np.exp(-self._factor / self._max_steps * (step - self._max_steps/2))))
    
    def noise(self):
        self._step += 1
        random = np.random.normal(loc=0, scale=self._get_noise(self._step))
        return np.ones(self.action_dimension) * random

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# and adapted to be synchronous with https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist