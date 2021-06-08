import argparse
import numpy as np
import torch
import wandb
import json
import os
import glob

from pyvirtualdisplay import Display

from components import Agent, Env
from utils import str2bool, save_uncert, init_uncert_file


def train_agent(env, eval_env, agent, nb_training_steps, nb_steps_target_replace, eval_episodes=3, eval_every=10, batch_size=256, checkpoint=60, skip_zoom=False):
    
    ob_t = env.reset()
    done = False
    episode_nb = 0
    episode_reward = 0
    episode_steps = 0
    eval_nb = 0
    running_score = 0


    for tr_step in range(nb_training_steps):

        if (tr_step + 1) % nb_steps_target_replace == 0:
            agent.replace_target_network()
        
        action, action_idx = agent.select_action(ob_t)
        
        ob_t1, reward, done = env.step(action)
        
        agent.store_transition(ob_t, action_idx, ob_t1, reward, done)

        if agent.nuber_experiences() > batch_size:
            agent.update()
            #agent.empty_buffer()
        
        ob_t = ob_t1
        
        episode_reward += reward
        episode_steps += 1

        if done:
            if episode_nb == 0:
                running_score = episode_reward
            else:
                running_score = running_score * 0.99 + episode_reward * 0.01
            print('Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4d | RunningScore: %4d | | Epsilon: %.3f' % \
                        (tr_step + 1, episode_nb + 1, episode_steps, episode_reward, running_score, agent._epsilon))
            wandb.log({'Train Episode': episode_nb + 1, 'Train Episode Score': float(episode_reward), "Train Episode Steps": episode_steps, "Train Running Score": running_score, 'Epsilon': agent._epsilon})
            episode_nb += 1
            ob_t = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            agent.epsilon_step()

            if skip_zoom:
                for _ in range(50):
                    ob_t, _, _, _ = env.step([0, 0, 0])
        
            if (episode_nb + 1) % eval_every == 0:
                mean_rwds, mean_steps = eval_agent(eval_env, agent, eval_nb, nb_episodes=eval_episodes)
                print('Evaluation Mean Steps: %4d | Mean Reward: %4d' % (mean_steps, mean_rwds))
                wandb.log({'Eval Episode': eval_nb + 1, 'Eval Episode Score': float(mean_rwds), "Eval Episode Steps": mean_steps})
                eval_nb += 1
            
            if (episode_nb + 1) % checkpoint == 0:
                agent.save_param(episode_nb)
            
            if running_score > env.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
                break

    agent.save_param(episode_nb)


def eval_agent(env, agent, eval_idx, nb_episodes=3):
    rewards = []
    total_steps = []

    for episode in range(nb_episodes):
        ob_t = env.reset()
        done = False
        episode_reward = 0
        nb_steps = 0

        while not done:
            
            action, _ = agent.select_action(ob_t, greedy=True)
        
            ob_t1, reward, done = env.step(action)
            
            ob_t = ob_t1
            episode_reward += reward
            nb_steps += 1

            if done:
                #print('Evaluation episode %3d | empty_vuSteps: %4d | Reward: %4d' % (episode + 1, nb_steps, episode_reward))
                rewards.append(episode_reward)
                total_steps.append(nb_steps)

                uncert = np.array([0] * (2*nb_steps))
                save_uncert(eval_idx, episode, episode_reward, uncert, file='uncertainties/train.txt', sigma=None)
    
    return np.mean(rewards), np.mean(total_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a DQN agent for the CarRacing-v0', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-EEv',
        '--eval-every', 
        type=int, 
        default=10, 
        help='Eval every n episodes')
    parser.add_argument(
        '-EEp',
        '--eval-episodes', 
        type=int, 
        default=3, 
        help='Number of evaluation episodes')
    parser.add_argument(
        '-ER',
        '--eval-render', 
        type=str2bool, 
        nargs='?',
        const=True,
        default=False, 
        help='Whether to render evaluation or not')
    parser.add_argument(
        '-IS',
        '--image-stack', 
        type=int, 
        default=3, 
        help='Number of images to stack')
    parser.add_argument(
        '-TS',
        '--train-seed', 
        type=float, 
        default=0, 
        help='Train Environment Random seed')
    parser.add_argument(
        '-ES',
        '--eval-seed', 
        type=float, 
        default=10, 
        help='Evaluation Environment Random seed')
    parser.add_argument(
        '-LR',
        '--learning-rate', 
        type=float, 
        default=0.001, 
        help='Learning Rate')
    parser.add_argument(
        '-G',
        '--gamma', 
        type=float, 
        default=0.95, 
        help='Discount factor')
    parser.add_argument(
        '-E',
        '--epsilon', 
        type=float, 
        default=1, 
        help='Epsilon greedy')
    parser.add_argument(
        '-ED',
        '--epsilon-decay', 
        type=float, 
        default=0.997, 
        help='Epsilon decay factor')
    parser.add_argument(
        '-EM',
        '--epsilon-min', 
        type=float, 
        default=0.1, 
        help='If epsilon decay, the minimum value of epsilon')
    parser.add_argument(
        '-NTS',
        '--training-steps', 
        type=int, 
        default=int(5e6), 
        help='Number traning steps')
    parser.add_argument(
        '-NTR',
        '--nb-target-replace', 
        type=int, 
        default=5000, 
        help='Number steps target network replace')
    parser.add_argument(
        '-BC',
        '--buffer-capacity', 
        type=int, 
        default=5000, 
        help='Replay buffer capacity')
    parser.add_argument(
        '-BS',
        '--batch-size', 
        type=int, 
        default=64, 
        help='Batch size')
    parser.add_argument(
        '-CR',
        '--clip-reward', 
        type=str, 
        default=None, 
        help='Clip reward')
    parser.add_argument(
        '-D',
        '--device', 
        type=str, 
        default='auto',
        help='Which device use: "cpu" or "cuda", "auto" for autodetect')
    args = parser.parse_args()
    
    old_settings = np.seterr(all='raise')

    actions = (
        [-1, 0, 0],              # Turn Left
        [1, 0, 0],               # Turn Right
        [0, 0, 1],              # Full Break
        [0, 1, 0],              # Accelerate
        [0, 0, 0],              # Do nothing

        [-1, 1, 0],             # Left accelerate
        [1, 1, 0],              # Right accelerate
        [-1, 0, 1],             # Left break
        [1, 0, 1],              # Right break
        
        [-0.5, 0, 0],           # Soft left
        [0.5, 0, 0],            # Soft right
        [0, 0, 0.5],            # Soft break
        [0, 0.5, 0],            # Soft accelerate

        [-0.5, 0.5, 0],         # Soft Left accelerate
        [0.5, 0.5, 0],          # Soft Right accelerate
        [-0.5, 0, 0.5],         # Soft Left break
        [0.5, 0, 0.5],          # Soft Right break
        )
    # actions = (
    #         [-1, 1, 0.2], 
    #         [0, 1, 0.2], 
    #         [1, 1, 0.2],
    #         [-1, 1,   0], 
    #         [0, 1,   0], 
    #         [1, 1,   0],
    #         [-1, 0, 0.2], 
    #         [0, 0, 0.2], 
    #         [1, 0, 0.2],
    #         [-1, 0,   0], 
    #         [0, 0,   0], 
    #         [1, 0,   0]
    #     )
    
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
        torch.manual_seed(args.train_seed)
        if use_cuda:
            torch.cuda.manual_seed(args.train_seed)
    else:
        device = args.device
    
    # Clip reward
    clip_reward = args.clip_reward.split(',') if args.clip_reward else None

    # Init Wandb
    with open("config.json") as config_file:
        config = json.load(config_file)
    wandb.init(project=config["project"], entity=config["entity"])

    # Init Agent and Environment
    agent = Agent(
        args.image_stack, 
        actions, 
        args.learning_rate, 
        args.gamma, 
        args.epsilon, 
        args.buffer_capacity, 
        args.batch_size, 
        device=device, 
        epsilon_decay=args.epsilon_decay, 
        clip_grad=True, 
        epsilon_min=args.epsilon_min)
    env = Env(
        img_stack=args.image_stack, 
        seed=args.train_seed, 
        clip_reward=clip_reward)
    eval_env = Env(
        img_stack=args.image_stack, 
        seed=args.eval_seed, 
        clip_reward=clip_reward,
        path_render='' if args.eval_render else None,
        validations=args.eval_episodes,
        evaluation=True)

    # Wandb config specification
    config = wandb.config
    config.args = args

    if isinstance(agent._net, list):
        wandb.watch(agent._net)
    else:
        wandb.watch(agent._net)

    train_agent(env, eval_env, agent, args.training_steps, args.nb_target_replace, eval_episodes=args.eval_episodes, eval_every=args.eval_every, batch_size=args.batch_size, checkpoint=60)