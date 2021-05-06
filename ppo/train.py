import argparse

import numpy as np
import torch
from utils import DrawLine, str2bool, save_uncert

import os
import time
#import sys
#import gc
#import psutil
#import nvidia_smi
#from pympler import asizeof

from agent import Agent
from env import Env

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, help='interval between training status logs (default: 10)')
parser.add_argument(
    '--val-interval', type=int, default=10, help='interval between evaluations (default: 10)')
parser.add_argument(
    '--from-checkpoint', type=str2bool, default=False, help='Whether to use checkpoint file (default: False)')
parser.add_argument(
    '----val-render', type=str2bool, default=False, help='render the environment on evaluation')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs (default: 2000)')
parser.add_argument(
    '--model', type=str, default='base', help='Type of uncertainty model (default: base)')
parser.add_argument(
    '--uncert-q', type=int, default=10, help='Number of networks to estimate uncertainties (default: 10)')
parser.add_argument(
    '--validations', type=int, default=1, help='Number validations each 10 epochs (default: 1)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

eval_max_ep = 1000

if __name__ == "__main__":
    print("Training model: {} with {} networks".format(args.model, args.uncert_q))
    agent = Agent(args, model=args.model)
    env = Env(args)
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")
    init_epoch = 0
    if args.from_checkpoint:
        init_epoch = agent.load_param()

    training_records = []
    running_score = 0
    state = env.reset()#save=True

    time_ep = []
    time_eval = 0

    #nvidia_smi.nvmlInit()
    #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    for i_ep in range(init_epoch, args.epochs):
        # process = psutil.Process(os.getpid())
        # print("Memory usage at {} epoch: {} GB".format(i_ep, process.memory_info().rss/1e9))
        # if process.memory_info().rss >= 1e10:   # if memory usage is over 10GB
        #     break
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        #res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

        score = 0
        state = env.reset()#load=True

        tic = time.time()
        for t in range(1000):
            action, a_logp, (_, _) = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        if len(time_ep) < 20:
            time_ep += [time.time() - tic]
        else:
            time_ep = time_ep[1:] + [time.time() - tic]


        if i_ep % args.log_interval == 0:
            if args.vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param(i_ep)        


        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break


        if i_ep % max(20, args.val_interval) == max(20, args.val_interval) - 1:
            l = np.mean(time_ep) * (args.epochs - i_ep - 1) + time_eval * (args.epochs - i_ep - 1)//args.val_interval
            h = l//3600
            m = l%3600//60
            s = l - h*3600 - m*60
            print("Left time: {:.0f}:{:.0f}:{:.0f}".format(h, m, s))


        if args.val_interval and i_ep % args.val_interval == 0:
            tic = time.time()
            mean_score = 0
            mean_uncert = np.array([0, 0], dtype=np.float64)
            for i_val in range(args.validations):
                #agent.eval_mode()
                score = 0
                state = env.reset()
                done = False

                uncert = []
                for t in range(eval_max_ep):
                    action, a_logp, (epis, aleat) = agent.select_action(state, eval=True)
                    uncert.append([epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]])
                    state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                    if args.val_render:
                        env.render()
                    score += reward
                    state = state_
                    if done or die:
                        break
                uncert = np.array(uncert)
                save_uncert(i_ep, i_val, score, uncert)

                mean_uncert += np.mean(uncert, axis=0) / args.validations
                mean_score += score / args.validations
            time_eval = time.time() - tic
            print("Eval score: {}".format(mean_score))
            print("Uncertainties: {}".format(mean_uncert))
            #agent.train_mode()s