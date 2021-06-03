import argparse

import numpy as np

import torch
import torch.nn as nn
import os
import glob
from pyvirtualdisplay import Display
from tqdm import tqdm
from collections import namedtuple, deque

from utilities import save_uncert, str2bool, init_uncert_file
from components import Env, Agent



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
        return self.memory


def test(env, agent, episodes, validations, add_uncert=False, img_stack=1):
    state = env.reset()
    memory = Memory(img_stack)

    for i_ep in tqdm(range(episodes)):
        mean_score = 0
        mean_uncert = np.array([0, 0], dtype=np.float64)
        mean_steps = 0
        uncert = None
        for i_val in range(validations):
            #agent.eval_mode()
            score = 0
            state = env.reset()

            # Add first state noise
            noisy_state = add_noise(state[0, :, :], std_dev[i_ep])
            [memory.push(noisy_state) for _ in range(img_stack)]
            state = np.stack(memory.sample()).squeeze()

            if add_uncert:
                env.plot_uncert(i_val, uncert, out_video=f'render/{i_val}.mp4')
            die = False
            steps = 0

            uncert = []
            while not die:
                steps += 1
                
                action, _, (epis, aleat) = agent.select_action(state, eval=True)
                uncert.append([epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]])
                state_, reward, _, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                score += reward

                # Add noise to next state and push into memory
                state_ = add_noise(state_[-1, :, :], std_dev[i_ep])
                memory.push(state_)
                state = np.stack(memory.sample()).squeeze()

            uncert = np.array(uncert)
            save_uncert(i_ep, i_val, score, uncert, sigma=std_dev[i_ep], file='uncertainties/test/test.txt')

            mean_uncert += np.mean(uncert, axis=0) / validations
            mean_score += score / validations
            mean_steps += steps / validations
        print('Ep {}\tScore: {:.2f}\tUncertainties: {}\tsigma: {:.2f}\tsteps: {:.1f}'.format(i_ep, mean_score, mean_uncert, std_dev[i_ep], mean_steps))


def add_noise(state, dev):
    noise = np.random.normal(loc=0, scale=dev, size=state.shape)
    noisy_state = state + noise
    noisy_state[noisy_state > 1] = 1
    noisy_state[noisy_state < -1] = -1
    return noisy_state



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the PPO agent for the CarRacing-v0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-IS',
        '--img-stack', 
        type=int, 
        default=4, 
        help='stack N images in a state')
    parser.add_argument(
        '-AR',
        '--action-repeat', 
        type=int, 
        default=8, 
        help='repeat action in N frames')
    parser.add_argument(
        '-S',
        '--seed', 
        type=int, 
        default=10, 
        help='random seed')
    parser.add_argument(
        '-R',
        '--render', 
        type=str2bool, 
        nargs='?',
        const=True,
        default=False, 
        help='render the environment')
    parser.add_argument(
        '-E',
        '--episodes', 
        type=int, 
        default=100, 
        help='Number of episodes to test')
    parser.add_argument(
        '-FC',
        '--from-checkpoint', 
        type=str, 
        default='param/ppo_net_params_base.pkl', 
        help='Path to trained model')
    parser.add_argument(
        '-M',
        '--model', 
        type=str, 
        default='base', 
        help='Type of uncertainty model: "base", "sensitivity", "dropout", "bootstrap", "mixture" or "bnn"')
    parser.add_argument(
        '-NN',
        '--nb-nets', 
        type=int, 
        default=10, 
        help='Number of networks to estimate uncertainties')
    parser.add_argument(
        '-V',
        '--validations', 
        type=int, 
        default=3, 
        help='Number of episodes to validate for each nose rate')
    parser.add_argument(
        '-T',
        '--std-thresh', 
        type=str, 
        default='0,0.5', 
        help='Upper and lower thresh for standar deviation noise grid, comma separated')
    parser.add_argument(
        '-AU',
        '--add-uncert', 
        type=str2bool, 
        nargs='?',
        const=True,
        default=False, 
        help='Whether to add uncertainties to render video or not')
    args = parser.parse_args()


    # Init folders and files to use
    if not os.path.exists('render'):
        os.makedirs('render')
    else:
        files = glob.glob('render/*')
        for f in files:
            os.remove(f)
    if not os.path.exists('uncertainties/test'):
        os.makedirs('uncertainties/test')
    init_uncert_file(file='uncertainties/test/test.txt')

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()


    # Whether to use cuda or cpu
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    args.gamma = 0
    # Init Agent and Environment
    agent = Agent(
        args.nb_nets, 
        args.img_stack,
        args.gamma,
        model=args.model)
    env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        path_render='' if args.render else None,
        validations=args.validations
    )
    agent.load_param(args.from_checkpoint)
    #agent.eval_mode()

    
    # Standar deviation grid
    thresh = [float(t) for t in args.std_thresh.split(',')]
    std_dev = np.linspace(thresh[0], thresh[1], num=args.episodes)

    test(env, agent, args.episodes, args.validations, add_uncert=args.add_uncert, img_stack=args.img_stack)