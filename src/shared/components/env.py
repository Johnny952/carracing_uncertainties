import imageio
import numpy as np
from gym.wrappers import Monitor
import gym

from shared.utils.noise import generate_noise_variance, add_noise


class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, img_stack, action_repeat, seed=0, path_render=None, validations=1, evaluation=False, noise=None):
        self.render_path = path_render is not None
        self.evaluation = evaluation
        if not self.render_path:
            self.env = gym.make('CarRacing-v0', verbose=0)
        else:
            self.validations = validations
            self.idx_val = validations // 2
            self.env = Monitor(gym.make('CarRacing-v0', verbose=0), './render/{}'.format(path_render),
                               video_callable=lambda episode_id: episode_id % validations == self.idx_val, force=True)
        self.env.seed(seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        #self.env._max_episode_steps = your_value

        # Noise in initial observations
        self.use_noise = False
        self.random_noise = 0
        if noise:
            if type(noise) is list:
                if len(noise) == 1:
                    self.set_noise_value(noise[0])
                elif len(noise) >= 2:
                    self.set_noise_range(noise)
            elif type(noise) is float and noise >= 0:
                self.set_noise_value(noise)
    
    def close(self):
        self.env.close()
    
    def set_noise_range(self, noise):
        assert type(noise) is list
        assert len(noise) >= 2
        self.use_noise = True
        self.generate_noise = True
        self.noise_lower, self.noise_upper = noise[0], noise[1]
    
    def set_noise_value(self, noise):
        assert noise >= 0
        self.use_noise = True
        self.generate_noise = False
        self.random_noise = noise

    def plot_uncert(self, index, uncertainties, width=56, out_video='render/test.mp4'):
        index = self.validations-1 if index == 0 else index-1
        if self.render_path and self.idx_val == index:
            max_unc = np.max(uncertainties, axis=0) + 1e-10

            # Append uncertainties to video
            vid = imageio.get_reader(self.env.videos[-1][0])
            fps = vid.get_meta_data()['fps']

            writer = imageio.get_writer(out_video, fps=fps)
            for idx, image in enumerate(vid.iter_data()):
                # action reapeat
                if idx > 0:
                    unct_idx = (idx-1) // self.action_repeat
                    bg = np.zeros((400, width, 3), dtype=np.uint8)
                    epist_height = int(
                        200 * uncertainties[unct_idx, 0] / max_unc[0])
                    bg[200:200+epist_height, :, :] = [255, 0, 0]
                    aleat_height = int(
                        200 * uncertainties[unct_idx, 1] / max_unc[1])
                    bg[:aleat_height, :, :] = [0, 0, 255]
                    image = np.concatenate((image, bg), axis=1)
                    writer.append_data(image)
            writer.close()

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)

        if self.use_noise:
            if self.generate_noise:
                self.random_noise = generate_noise_variance(
                    self.noise_lower, self.noise_upper)
            img_gray = add_noise(img_gray, self.random_noise)

        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for _ in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        # Add noise in observation
        if self.use_noise:
            img_gray = add_noise(img_gray, self.random_noise)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        return self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


def make_env(img_stack, action_repeat, seed=0, path_render=None, validations=1, evaluation=False):
    def fn():
        env = Env(
            img_stack,
            action_repeat,
            seed=seed,
            path_render=path_render,
            validations=validations,
            evaluation=evaluation
        )
        return env
    return fn
