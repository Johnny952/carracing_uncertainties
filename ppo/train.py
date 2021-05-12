import argparse
import numpy as np
import torch
import os
import time
import wandb
import json
import glob

from utilities import str2bool, save_uncert, init_uncert_file
from agent import Agent
from env import Env

from pyvirtualdisplay import Display


def train_agent(agent, env, eval_env, episodes, nb_validations=1, init_ep=0, log_interval=10, val_interval=10, val_render=False):
    running_score = 0
    state = env.reset()

    time_ep = []
    time_eval = 0
    eval_idx = 0

    for i_ep in range(init_ep, episodes):
        score = 0
        state = env.reset()#load=True

        tic = time.time()
        for t in range(1000):
            action, a_logp, (_, _) = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if agent.store_transition((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_

            #wandb.log({'Step Reward': float(reward), 'Step Score': float(score)})

            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        wandb.log({'Episode': i_ep, 'Episode Running Score': float(running_score), 'Episode Score': float(score)})

        if len(time_ep) < 20:
            time_ep += [time.time() - tic]
        else:
            time_ep = time_ep[1:] + [time.time() - tic]


        if i_ep % log_interval == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param(i_ep)


        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break


        if i_ep % max(20, val_interval) == max(20, val_interval) - 1:
            l = np.mean(time_ep) * (episodes - i_ep - 1) + time_eval * (episodes - i_ep - 1)//val_interval
            h = l//3600
            m = l%3600//60
            s = l - h*3600 - m*60
            print("Estimated finish time: {:.0f}:{:.0f}:{:.0f}".format(h, m, s))


        if val_interval and i_ep % val_interval == 0:
            tic = time.time()
            mean_score, mean_uncert = eval_agent(agent, eval_env, nb_validations, i_ep, render=val_render)
            wandb.log({'Idx': eval_idx, 'Eval Mean Score': float(mean_score), 'Eval Mean Epist Uncert': float(mean_uncert[0]), 'Eval Mean Aleat Uncert': float(mean_uncert[0])})
            time_eval = time.time() - tic
            eval_idx += 1
            print("Eval score: {}".format(mean_score))
            print("Uncertainties: {}".format(mean_uncert))
            #agent.train_mode()


def eval_agent(agent, env, validations, epoch, render=False):
    eval_max_ep = 1000

    mean_score = 0
    mean_uncert = np.array([0, 0], dtype=np.float64)
    for i_val in range(validations):
        #agent.eval_mode()
        score = 0
        state = env.reset()
        done = False
        die = False

        uncert = []
        idx = 0
        while not die:
            action, a_logp, (epis, aleat) = agent.select_action(state, eval=True)
            uncert.append([epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]])
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            score += reward
            state = state_

        uncert = np.array(uncert)
        save_uncert(epoch, i_val, score, uncert)

        mean_uncert += np.mean(uncert, axis=0) / validations
        mean_score += score / validations

    return mean_score, mean_uncert




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a PPO agent for the CarRacing-v0', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-G',
        '--gamma', 
        type=float, 
        default=0.99, 
        help='discount factor')
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
        default=0, 
        help='random seed')
    parser.add_argument(
        '-ES',
        '--eval-seed', 
        type=int, 
        default=1, 
        help='random evaluation environment seed')
    parser.add_argument(
        '-LI',
        '--log-interval', 
        type=int, 
        default=10, 
        help='Interval between training status logs')
    parser.add_argument(
        '-VI',
        '--val-interval', 
        type=int, 
        default=10, 
        help='Interval between evaluations')
    parser.add_argument(
        '-FC',
        '--from-checkpoint', 
        type=str2bool, 
        default=False, 
        help='Whether to use checkpoint file')
    parser.add_argument(
        '-VR',
        '--val-render', 
        type=str2bool, 
        default=False, 
        help='render the environment on evaluation')
    parser.add_argument(
        '-E',
        '--epochs', 
        type=int, 
        default=2000, 
        help='Number of epochs')
    parser.add_argument(
        '-M',
        '--model', 
        type=str, 
        default='base', 
        help='Type of uncertainty model (default: base)')
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
        help='Number validations each 10 epochs')
    args = parser.parse_args()


    # Init model checkpoint folder and uncertainties folder
    if not os.path.exists('param'):
        os.makedirs('param')
    if not os.path.exists('uncertainties'):
        os.makedirs('uncertainties')
    if not os.path.exists('render'):
        os.makedirs('render')
    else:
        files = glob.glob('render/*')
        for f in files:
            os.remove(f)

        
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

    # Init Wandb
    with open("config.json") as config_file:
        config = json.load(config_file)
    wandb.init(project=config["project"], entity=config["entity"])

    #print("Training model: {} with {} networks".format(args.model, args.uncert_q))
    
    # Init Agent and Environment
    agent = Agent(
        args.nb_nets, 
        args.img_stack,
        args.gamma,
        from_checkpoint=args.from_checkpoint,
        model=args.model)
    env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.seed
    )
    eval_env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.eval_seed,
        path_render='' if args.val_render else None
    )
    init_epoch = 0
    if args.from_checkpoint:
        init_epoch = agent.load_param()
    
    # Wandb config specification
    config = wandb.config
    config.learning_rate = agent.lr
    config.batch_size = agent.batch_size
    config.max_grad_norm = agent.max_grad_norm
    config.clip_param = agent.clip_param
    config.ppo_epoch = agent.ppo_epoch
    config.buffer_capacity = agent.buffer_capacity
    config.args = args

    init_uncert_file()

    if isinstance(agent._model._model, list):
        wandb.watch(agent._model._model[0])
    else:
        wandb.watch(agent._model._model)

    train_agent(
        agent, env, eval_env, 
        episodes=args.epochs, 
        nb_validations=args.validations, 
        init_ep=init_epoch,  
        log_interval=args.log_interval, 
        val_interval=args.val_interval, 
        val_render=args.val_render)
