import argparse
import numpy as np
import torch
import os
import wandb
import json
import glob
from tqdm import tqdm

from utilities import str2bool, save_uncert, init_uncert_file
from components import Agent, make_env, SubprocVecEnv


from pyvirtualdisplay import Display


def make_mp_envs(num_env, agent, img_stack, action_repeat, seed=0, path_render=None, validations=2, evaluation=False):
    return SubprocVecEnv([make_env(img_stack, action_repeat, seed=seed+i, path_render=path_render, validations=validations, evaluation=evaluation) for i in range(num_env)], agent)

def train_agent(nb_processes, agent, env, eval_env, episodes, nb_validations=1, init_ep=0, log_interval=10, val_interval=10, val_render=False):
    running_score = 0
    eval_idx = 0

    for i_ep in range(init_ep, episodes//nb_processes):
        scores, steps, _ = env.rollout()

        for j, (score, step) in enumerate(zip(scores, steps)):
            running_score = running_score * 0.99 + scores[j] * 0.01
            wandb.log({
                'Train Episode': i_ep + j, 
                'Episode Running Score': float(running_score), 
                'Episode Score': float(score),
                'Episode Steps': float(step)
            })
        
        if agent.able_sample():
            print('updating')
            agent.update()
        
        if i_ep % log_interval//nb_processes == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, scores[-1], running_score))
            agent.save_param(i_ep)
        
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, scores[-1]))
            break
    
        if i_ep % val_interval//nb_processes == 0:
            mean_score, mean_uncert, mean_steps = eval_agent(nb_processes, eval_env, nb_validations, i_ep)
            wandb.log({
                'Eval Episode': eval_idx, 
                'Eval Mean Score': float(mean_score), 
                'Eval Mean Epist Uncert': float(mean_uncert[0]), 
                'Eval Mean Aleat Uncert': float(mean_uncert[1]), 
                'Eval Mean Steps': float(mean_steps)
                })
            eval_idx += 1
            print("Eval score: {}\tSteps: {}\tUncertainties: {}".format(mean_score, mean_steps, mean_uncert))
    
    env.close()
    eval_env.close()



def eval_agent(nb_processes, env, validations, epoch):
    mean_score = 0
    mean_uncert = np.array([0, 0], dtype=np.float64)
    mean_steps = 0

    for i_val in range(validations//nb_processes):
        scores, steps, _ = env.rollout()

        for j, (score, step) in enumerate(zip(scores, steps)):
            save_uncert(epoch, i_val, score, uncert, file='uncertainties/train/train.txt')

            mean_uncert += np.mean(uncert, axis=0) / validations
            mean_score += score / validations
            mean_steps += steps / validations

    return mean_score, mean_uncert, mean_steps

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a PPO agent for the CarRacing-v0 in multiprocessing environment', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-NP',
        '--nb-processes', 
        type=int, 
        default=2, 
        help='Number of subprocesses')
    parser.add_argument(
        '-G',
        '--gamma', 
        type=float, 
        default=0.99, 
        help='Discount factor')
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
        nargs='?',
        const=True,
        default=False, 
        help='Whether to use checkpoint file')
    parser.add_argument(
        '-VR',
        '--val-render', 
        type=str2bool, 
        nargs='?',
        const=True,
        default=False, 
        help='render the environment on evaluation')
    parser.add_argument(
        '-E',
        '--epochs', 
        type=int, 
        default=2000, 
        help='Number of epochs, divisible by 2')
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
        default=4,
        help='Number validations each 10 epochs, divisible by 2')
    parser.add_argument(
        '-D',
        '--device', 
        type=str, 
        default='auto',
        help='Which device use: "cpu" or "cuda", "auto" for autodetect')
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
    if not os.path.exists('uncertainties/train'):
        os.makedirs('uncertainties/train')
    init_uncert_file(file='uncertainties/train/train.txt')

        
    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Whether to use cuda or cpu
    if args.device == 'auto':
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)
    else:
        device = args.device

    # Init Wandb
    with open("config.json") as config_file:
        config = json.load(config_file)
    wandb.init(project=config["project"], entity=config["entity"])

    # Enable torch multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)


    # Init Agent and Environment
    agent = Agent(
        args.nb_nets, 
        args.img_stack,
        args.gamma,
        model=args.model,
        device=device)
    env = make_mp_envs(
        args.nb_processes, 
        agent, 
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.seed
    )
    eval_env = make_mp_envs(
        args.nb_processes, 
        agent, 
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        path_render='' if args.val_render else None,
        validations=args.validations,
        evaluation=True
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
    config.device = agent.device
    config.args = args

    if isinstance(agent._model._model, list):
        wandb.watch(agent._model._model[0])
    else:
        wandb.watch(agent._model._model)
    

    train_agent(
        args.nb_processes, agent, env, eval_env, 
        episodes=args.epochs, 
        nb_validations=args.validations, 
        init_ep=init_epoch,  
        log_interval=args.log_interval, 
        val_interval=args.val_interval, 
        val_render=args.val_render)