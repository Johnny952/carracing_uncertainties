import gym
import numpy as np
from collections import deque
from statistics import mean

from utils import imgstackRGB2graystack


class Env():
    def __init__(self, img_stack=4, seed=0, clip_reward=None):
        """Environment Constructor

        Args:
            img_stack (int, optional): Number of consecutive frames to stack. Defaults to 4.
            seed (int, optional): Random seed env generator. Defaults to 0.
        """        
        env = gym.make('CarRacing-v0')
        self._env = gym.wrappers.FrameStack(env, img_stack)
        self._env.seed(seed)

        self.low_state = self._env.action_space.low
        self.high_state = self._env.action_space.high
        self._reward_memory = deque([])

        self._clip_reward=clip_reward

    def reset(self):
        """Resets the environment

        Returns:
            np.ndarray: Last n gray frames stack
        """        
        self._reward_memory.clear()
        return imgstackRGB2graystack(self._env.reset())
    
    def step(self, action):
        """Step in the environment. Transition to the next state.

        Args:
            action (array): 3 dimensional array, first dimension steering angle in range [-1, 1], second dimension throttle in range [0, 1] and last dimensino brake in range [0, 1]

        Returns:
            np.ndarray: Last n gray frames stack finishing in time t+1
            float: Reward of the transition t to t+1
            done: Whether the episode is finished or not
        """        
        next_state, reward, done, _ = self._env.step(action)
        # green penalty last state
        if np.mean(next_state[-1][:, :, 1]) > 185.0:
            reward -= 5
        
        # penalty for die state
        if len(self._reward_memory) > 20 and sum(self._reward_memory) <= -20:
            done = True
            reward -= 20
        
        if self._clip_reward is not None:
            reward = np.clip(reward, a_max=self._clip_reward)

        # push reward in memory
        self._reward_memory.append(reward)
        
        return imgstackRGB2graystack(next_state), reward, done
    
    def render(self, *args):
        """Show the state of the environment"""        
        self._env.render(*args)


if __name__ == "__main__":
    env = Env()

    s0 = env.reset()

    for i in range(100):
        next_state, reward, done = env.step([0, 1, 0])
        if done:
            break
        print("Step: {:.0f}, Reward: {:.3f}, Done: {}".format(i+1, reward, done))
        env.render()
    print("")