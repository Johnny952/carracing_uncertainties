import argparse
import numpy as np
import torch
import os
import wandb
import json
import glob
from tqdm import tqdm
from termcolor import colored

from utilities import save_uncert, init_uncert_file
from components import Agent, Env

from pyvirtualdisplay import Display


def train_agent(agent, env, eval_env, episodes, nb_validations=1, init_ep=0, log_interval=10, val_interval=10):
    running_score = 0
    state = env.reset()

    eval_idx = 0
    best_score = 0

    for i_ep in tqdm(range(init_ep, episodes)):
        score = 0
        steps = 0
        state = env.reset()

        for _ in range(1000):
            action, a_logp = agent.select_action(state)[:2]
            state_, reward, done, die = env.step(
                action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if agent.store_transition((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            steps += 1

            #wandb.log({'Step Reward': float(reward), 'Step Score': float(score)})

            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        wandb.log({
            'Train Episode': i_ep,
            'Episode Running Score': float(running_score),
            'Episode Score': float(score),
            'Episode Steps': float(steps),
        })

        if i_ep % log_interval == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(
                i_ep, score, running_score))
            agent.save(i_ep)

        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(
                running_score, score))
            break

        if val_interval and i_ep % val_interval == 0:
            mean_score, mean_uncert, mean_steps = eval_agent(
                agent, eval_env, nb_validations, i_ep)
            wandb.log({
                'Eval Episode': eval_idx,
                'Eval Mean Score': float(mean_score),
                'Eval Mean Epist Uncert': float(mean_uncert[0]),
                'Eval Mean Aleat Uncert': float(mean_uncert[1]),
                'Eval Mean Steps': float(mean_steps)
            })
            if mean_score > best_score:
                agent.save(i_ep, path='param/best_agent.pkl')
                best_score = mean_score

            eval_idx += 1
            print("Eval score: {}\tSteps: {}\tUncertainties: {}".format(
                mean_score, mean_steps, mean_uncert))
            # agent.train_mode()


def eval_agent(agent, env, validations, epoch):
    mean_score = 0
    mean_uncert = np.array([0, 0], dtype=np.float64)
    mean_steps = 0
    for i_val in range(validations):
        # agent.eval_mode()
        score = 0
        steps = 0
        state = env.reset()
        die = False

        uncert = []
        while not die:
            action, _, (epis, aleat) = agent.select_action(
                state, eval=True)
            uncert.append([epis.view(-1).cpu().numpy()[0],
                          aleat.view(-1).cpu().numpy()[0]])
            state_, reward, _, die = env.step(
                action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            score += reward
            state = state_
            steps += 1

        uncert = np.array(uncert)
        save_uncert(epoch, i_val, score, uncert,
                    file='uncertainties/train/train.txt')

        mean_uncert += np.mean(uncert, axis=0) / validations
        mean_score += score / validations
        mean_steps += steps / validations

    return mean_score, mean_uncert, mean_steps


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
        default=5000,
        help='Interval between training status logs')
    parser.add_argument(
        '-VI',
        '--val-interval',
        type=int,
        default=20,
        help='Interval between evaluations')
    parser.add_argument(
        '-FC',
        '--from-checkpoint',
        action='store_true',
        help='Whether to use checkpoint file')
    parser.add_argument(
        '-VR',
        '--val-render',
        action='store_true',
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
        help='Type of uncertainty model: "base", "sensitivity", "dropout", "bootstrap" or "bnn"')
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
    parser.add_argument(
        '-D',
        '--device',
        type=str,
        default='auto',
        help='Which device use: "cpu" or "cuda", "auto" for autodetect')
    parser.add_argument(
        '-N',
        '--noise',
        type=str,
        default='0,0.3',
        # default=None,
        help='Whether to use noise or not, and standard deviation bounds separated by comma (ex. "0,0.5")')
    args = parser.parse_args()

    print(colored('Initializing data folders', 'blue'))
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
    print(colored('Data folders created successfully', 'green'))

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

    #print("Training model: {} with {} networks".format(args.model, args.uncert_q))

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(',')]
        print_arg = f'noisy observation with {args.noise} std bounds'
    else:
        add_noise = None
        print_arg = ''

    # Init Agent and Environment
    print(colored('Initializing agent and environments', 'blue'))
    agent = Agent(
        args.nb_nets,
        args.img_stack,
        args.gamma,
        model=args.model,
        device=device)
    env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        noise=add_noise,
    )
    eval_env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.eval_seed,
        path_render='' if args.val_render else None,
        validations=args.validations,
        evaluation=True,
        noise=add_noise,
    )
    init_epoch = 0
    if args.from_checkpoint:
        init_epoch = agent.load()
    print(colored('Agent and environments created successfully', 'green'))

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

    print(colored(
        f'Training {args.model} during {args.epochs} epochs and {print_arg}', 'magenta'))

    train_agent(
        agent, env, eval_env,
        episodes=args.epochs,
        nb_validations=args.validations,
        init_ep=init_epoch,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
    )
