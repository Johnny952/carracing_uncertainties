import gym
from gym.wrappers import Monitor
import numpy as np
from collections import deque

from utils import imgstackRGB2graystack


class Env():
    def __init__(self, img_stack=4, seed=0, clip_reward=None, path_render=None, validations=1, evaluation=False, action_repeat=1):
        """Environment Constructor

        Args:
            img_stack (int, optional): Number of consecutive frames to stack. Defaults to 4.
            seed (int, optional): Random seed env generator. Defaults to 0.
        """
        self._action_repeat = action_repeat
        self.render = path_render is not None
        self.evaluation = evaluation
        env = gym.make('CarRacing-v0')
        self._env = gym.wrappers.FrameStack(env, img_stack)
        if self.render:
            self.validations = validations
            self.idx_val = validations // 2
            self._env = Monitor(self._env, './render/{}'.format(path_render), video_callable=lambda episode_id: episode_id%validations==self.idx_val, force=True)    
        self._env.seed(seed)

        self.low_state = self._env.action_space.low
        self.high_state = self._env.action_space.high
        self._reward_memory = deque([], maxlen=50)

        self.reward_threshold = self._env.spec.reward_threshold

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
        reward = 0
        for _ in range(self._action_repeat):
            next_state, rwd, done, _ = self._env.step(action)
            # green penalty last state
            if np.mean(next_state[-1][:, :, 1]) > 185.0:
                rwd -= 0.05
            
            # reward for full gas
            if action[1] == 1 and action[2] == 0:
                rwd += 1.5*np.abs(rwd)
            
            if self._clip_reward is not None:
                rwd = np.clip(rwd, a_max=self._clip_reward)

            # push reward in memory
            # self._reward_memory.append(reward)

            # penalty for die state
            # die = sum(self._reward_memory) <= -5
            # if not self.evaluation and die:
            #     done = True
            #     #reward -= 20
            #     reward += 100
            reward += rwd
            if done:
                break
        
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