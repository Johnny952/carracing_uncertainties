import argparse
import numpy as np
import torch
import wandb
import json
import os
import glob

from pyvirtualdisplay import Display

from components import make_agent, Env
from utils import str2bool, save_uncert, init_uncert_file


def train_agent(env, eval_env, agent, nb_training_ep, nb_steps_target_replace, eval_episodes=3, eval_every=10, updates_after=1000, update_each=256, skip_zoom=None):
    
    ob_t = env.reset()
    done = False
    episode_nb = 0
    episode_reward = 0
    episode_steps = 0
    eval_nb = 0
    running_score = 0
    tr_step = 0
    best_eval_score = -100

    if skip_zoom is not None:
        for _ in range(skip_zoom):
            ob_t, _, _ = env.step([0, 0, 0])


    while episode_nb < nb_training_ep:       
        action, action_idx = agent.select_action(ob_t)
        
        ob_t1, reward, done = env.step(action)
        
        agent.store_transition(ob_t, action_idx, ob_t1, reward, done)

        if tr_step % update_each == 0 and agent.number_experiences() >= updates_after:
            #print("Updating")
            agent.update_minibatch()
            #agent.update()
            #agent.epsilon_step()
            #agent.empty_buffer()
        
        ob_t = ob_t1
        
        episode_reward += reward
        episode_steps += 1

        if done:
            running_score = episode_reward if episode_nb == 0 else running_score * 0.99 + episode_reward * 0.01
            
            if episode_nb % nb_steps_target_replace == 0:
                agent.replace_target_network()

            print('Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4d | RunningScore: %4d | | Epsilon: %.3f' % \
                        (tr_step + 1, episode_nb + 1, episode_steps, episode_reward, running_score, agent.epsilon()))
            wandb.log({'Train Episode': episode_nb + 1, 'Train Episode Score': float(episode_reward), "Train Episode Steps": episode_steps, "Train Running Score": running_score, 'Epsilon': agent.epsilon()})
            episode_nb += 1
            ob_t = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            agent.epsilon_step()

            if skip_zoom is not None:
                for _ in range(skip_zoom):
                    ob_t, _, _ = env.step([0, 0, 0])
        
            if (episode_nb + 1) % eval_every == 0:
                mean_rwds, mean_steps = eval_agent(eval_env, agent, eval_nb, nb_episodes=eval_episodes)
                print('Evaluation Mean Steps: %4d | Mean Reward: %4d' % (mean_steps, mean_rwds))
                wandb.log({'Eval Episode': eval_nb + 1, 'Eval Episode Score': float(mean_rwds), "Eval Episode Steps": mean_steps})
                eval_nb += 1
                if mean_rwds >= best_eval_score:
                    agent.save_param(episode_nb)
            
            if running_score > env.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
                break
        tr_step += 1

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
        '-EM',
        '--epsilon-method', 
        type=str, 
        default='linear', 
        help='Epsilon decay method: constant, linear, exp or inverse_sigmoid')
    parser.add_argument(
        '-EMa',
        '--epsilon-max', 
        type=float, 
        default=1, 
        help='The minimum value of epsilon, used this value in constant')
    parser.add_argument(
        '-EMi',
        '--epsilon-min', 
        type=float, 
        default=0.1, 
        help='The minimum value of epsilon')
    parser.add_argument(
        '-EF',
        '--epsilon-factor', 
        type=float, 
        default=3, 
        help='Factor parameter of epsilon decay, only used when method is exp or inverse_sigmoid')
    parser.add_argument(
        '-EMS',
        '--epsilon-max-steps', 
        type=int, 
        default=500, 
        help='Max Epsilon Steps parameter, when epsilon is close to the minimum')
    parser.add_argument(
        '-NTS',
        '--training-ep', 
        type=int, 
        default=1000, 
        help='Number traning episodes')
    parser.add_argument(
        '-NTR',
        '--nb-target-replace', 
        type=int, 
        default=5, 
        help='Number episodes target network replace, only used when model is dqn')
    parser.add_argument(
        '-BC',
        '--buffer-capacity', 
        type=int, 
        default=2000, 
        help='Replay buffer capacity')
    parser.add_argument(
        '-BS',
        '--batch-size', 
        type=int, 
        default=128, 
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
    parser.add_argument(
        '-AR',
        '--action-repeat', 
        type=int, 
        default=1, 
        help='Number steps using same action')
    parser.add_argument(
        '-UE',
        '--update-each', 
        type=int, 
        default=2000, 
        help='Updates every number of steps')
    parser.add_argument(
        '-SZ',
        '--skip-zoom', 
        type=int, 
        default=2, 
        help='Number of steps to skip at episode start')
    parser.add_argument(
        '-M',
        '--model', 
        type=str, 
        default='dqn', 
        help='Which RL model use: dqn, ddqn2015 or ddqn2018')
    parser.add_argument(
        '-T',
        '--tau', 
        type=float, 
        default=0.1, 
        help='DDQN Tau parameter, only used when model is ddqn2015')
    args = parser.parse_args()
    
    old_settings = np.seterr(all='raise')

    # actions = (
    #     [-1, 0, 0],              # Turn Left
    #     [1, 0, 0],               # Turn Right
    #     [0, 0, 1],              # Full Break
    #     [0, 1, 0],              # Accelerate
    #     [0, 0, 0],              # Do nothing

    #     [-1, 1, 0],             # Left accelerate
    #     [1, 1, 0],              # Right accelerate
    #     [-1, 0, 1],             # Left break
    #     [1, 0, 1],              # Right break
        
    #     [-0.5, 0, 0],           # Soft left
    #     [0.5, 0, 0],            # Soft right
    #     [0, 0, 0.5],            # Soft break
    #     [0, 0.5, 0],            # Soft accelerate

    #     [-0.5, 0.5, 0],         # Soft Left accelerate
    #     [0.5, 0.5, 0],          # Soft Right accelerate
    #     [-0.5, 0, 0.5],         # Soft Left break
    #     [0.5, 0, 0.5],          # Soft Right break
    #     )
    actions = (
            [-1, 1, 0.2], 
            [0, 1, 0.2], 
            [1, 1, 0.2],
            [-1, 1,   0], 
            [0, 1,   0], 
            [1, 1,   0],
            [-1, 0, 0.2], 
            [0, 0, 0.2], 
            [1, 0, 0.2],
            [-1, 0,   0], 
            [0, 0,   0], 
            [1, 0,   0]
        )
    
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
    agent = make_agent(
        args.model,
        args.image_stack, 
        actions, 
        args.learning_rate, 
        args.gamma, 
        args.training_ep,
        args.buffer_capacity, 
        args.batch_size, 
        device=device, 
        clip_grad=False, 
        epsilon_method=args.epsilon_method,
        epsilon_max=args.epsilon_max,
        epsilon_min=args.epsilon_min,
        epsilon_factor=args.epsilon_factor,
        epsilon_max_steps=args.epsilon_max_steps,
        tau=args.tau)
    env = Env(
        img_stack=args.image_stack, 
        seed=args.train_seed, 
        clip_reward=clip_reward,
        action_repeat=args.action_repeat)
    eval_env = Env(
        img_stack=args.image_stack, 
        seed=args.eval_seed, 
        clip_reward=clip_reward,
        path_render='' if args.eval_render else None,
        validations=args.eval_episodes,
        evaluation=True,
        action_repeat=args.action_repeat)

    # Wandb config specification
    config = wandb.config
    config.args = args

    
    if args.model.lower() in ['dqn', 'ddqn2015']:
        wandb.watch(agent._model)
    elif args.model.lower() in ['ddqn2018']:
        wandb.watch(agent._model1)

    train_agent(
        env, eval_env, agent, 
        args.training_ep,
        args.nb_target_replace, 
        eval_episodes=args.eval_episodes, 
        eval_every=args.eval_every, 
        updates_after=args.batch_size, 
        update_each=args.update_each,
        skip_zoom=args.skip_zoom)
    
    env.close()
    eval_env.close()