import argparse

import numpy as np

import gym
import torch
import torch.nn as nn

from utils import save_uncert

from env import Env
from agent import Agent

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to test')
parser.add_argument('--from-checkpoint', type=str, default='param/ppo_net_params_base.pkl', help='Path to trained model')
parser.add_argument(
    '--model', type=str, default='base', help='Type of uncertainty model (default: base)')
parser.add_argument(
    '--uncert-q', type=int, default=10, help='Number of networks to estimate uncertainties (default: 10)')
parser.add_argument(
    '--validations', type=int, default=2, help='Number of episodes to validate for each nose rate (default: 2)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

args.gamma = 0

thresh = [0, 0.5]
std_dev = np.linspace(thresh[0], thresh[1], num=args.episodes)

def add_noise(state, dev):
    noise = np.random.normal(loc=0, scale=dev, size=state.shape)
    noisy_state = state + noise
    noisy_state[noisy_state > 1] = 1
    noisy_state[noisy_state < -1] = -1
    return noisy_state

if __name__ == "__main__":
    agent = Agent(args, model=args.model)
    agent.load_param()
    #agent.eval_mode()
    env = Env(args)

    state = env.reset()

    for i_ep in range(args.episodes):
        mean_score = 0
        mean_uncert = np.array([0, 0], dtype=np.float64)

        for i_val in range(args.validations):
            #agent.eval_mode()
            score = 0
            state = env.reset()
            done = False

            uncert = []
            for t in range(1000):
                state = add_noise(state, std_dev[i_ep])
                action, a_logp, (epis, aleat) = agent.select_action(state, eval=True)
                uncert.append([epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]])
                state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                if args.render:
                    env.render()
                score += reward
                state = state_
                if done or die:
                    break
            uncert = np.array(uncert)
            save_uncert(i_ep, i_val, score, uncert, file='uncertainties/test.txt', sigma=std_dev[i_ep])

            mean_uncert += np.mean(uncert, axis=0) / args.validations
            mean_score += score / args.validations
        print('Ep {}\tScore: {:.2f}\tUncertainties: {}\tsigma= {:.2f}'.format(i_ep, mean_score, mean_uncert, std_dev[i_ep]))