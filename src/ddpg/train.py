import argparse
import numpy as np
import torch
import wandb
import os
import glob
from termcolor import colored
from pyvirtualdisplay import Display
import uuid

import sys
sys.path.append('..')
from shared.utils.utils import init_uncert_file
from components.agent import Agent
from components.trainer import Trainer
from utilities.noise import OUNoise, BaseNoise
from shared.components.env import Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DDPG agent for CarRacing-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment Config
    env_config = parser.add_argument_group("Environment config")
    env_config.add_argument(
        "-AR",
        "--action-repeat",
        type=int,
        default=8,
        help="Number steps using same action",
    )
    env_config.add_argument(
        "-N",
        "--noise",
        type=str,
        default="0,0.1",
        # default=None,
        help='Whether to use noise or not, and standard deviation bounds separated by comma (ex. "0,0.5")',
    )
    env_config.add_argument(
        "-TS",
        "--train-seed",
        type=float,
        default=0,
        help="Train Environment Random seed",
    )
    env_config.add_argument(
        "-ES",
        "--eval-seed",
        type=float,
        default=10,
        help="Evaluation Environment Random seed",
    )
    env_config.add_argument(
        "-GR",
        "--green-reward",
        type=float,
        default=-0.1,
        help="Penalization for observation with green color",
    )
    env_config.add_argument(
        "-DR",
        "--done-reward",
        type=float,
        default=-100,
        help="Penalization for ending episode because of low reward",
    )

    # Agent Config
    agent_config = parser.add_argument_group("Agent config")
    agent_config.add_argument(
        "-ALR",
        "--actor-lr",
        type=float,
        default=1e-4,
        help="Actor learning rate",
    )
    agent_config.add_argument(
        "-CLR",
        "--critic-lr",
        type=float,
        default=1e-3,
        help="Critic learning rate",
    )
    agent_config.add_argument(
        "-CWD",
        "--critic-wd",
        type=float,
        default=1e-2,
        help="Critic weight decay",
    )
    agent_config.add_argument(
        "-T",
        "--tau",
        type=float,
        default=1e-3,
        help="Soft update parameter",
    )
    agent_config.add_argument(
        "-G", "--gamma", type=float, default=0.99, help="Discount factor"
    )
    agent_config.add_argument(
        "-IS", "--image-stack", type=int, default=4, help="Number of images to stack"
    )
    agent_config.add_argument(
        "-BC",
        "--buffer-capacity",
        type=int,
        default=1e6,
        help="Replay buffer capacity",
    )
    agent_config.add_argument(
        "-BS", "--batch-size", type=int, default=32, help="Batch size"
    )
    agent_config.add_argument(
        "-FC", "--from-checkpoint", type=str, default=None, help="Path to trained model"
    )
    agent_config.add_argument(
        "-NU", "--nb-updates", type=int, default=100, help="Number of agent updates"
    )

    # Training Config
    train_config = parser.add_argument_group("Train config")
    train_config.add_argument(
        "-ER",
        "--eval-render",
        action="store_true",
        help="Whether to render evaluation or not",
    )
    train_config.add_argument(
        "-NTS", "--training-ep", type=int, default=3000, help="Number traning episodes"
    )
    train_config.add_argument(
        "-SZ",
        "--skip-zoom",
        type=int,
        default=0,
        help="Number of steps to skip at episode start",
    )
    train_config.add_argument(
        "-EEv", "--eval-every", type=int, default=20, help="Eval every n episodes"
    )
    train_config.add_argument(
        "-EEp",
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes",
    )
    train_config.add_argument(
        "-UE",
        "--update-every",
        type=int,
        default=2000,
        help="Update agent every n steps",
    )
    train_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    args = parser.parse_args()

    run_name = f"ddpg_{uuid.uuid4()}"
    
    old_settings = np.seterr(all="raise")

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    if not os.path.exists("param"):
        os.makedirs("param")
    if not os.path.exists("uncertainties"):
        os.makedirs("uncertainties")
    if not os.path.exists("render"):
        os.makedirs("render")
    if not os.path.exists(f"render/{run_name}") and args.eval_render:
        os.makedirs(f"render/{run_name}")
    else:
        files = glob.glob(f"render/{run_name}/*")
        for f in files:
            os.remove(f)
    if not os.path.exists("uncertainties/train"):
        os.makedirs("uncertainties/train")
    init_uncert_file(file=f"uncertainties/train/{run_name}.txt")
    print(colored("Data folders created successfully", "green"))

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Whether to use cuda or cpu
    if args.device == "auto":
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.train_seed)
        if use_cuda:
            torch.cuda.manual_seed(args.train_seed)
    else:
        device = args.device
    print(colored(f"Using: {device}", "green"))

    # Init Wandb
    wandb.init(project='carracing-ddpg')

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    action_dim = 3
    noises = [
        BaseNoise(1, 40*args.training_ep//2, max_=0.6, min_=0.01),
        BaseNoise(1, 40*args.training_ep//2, max_=0.3, min_=0.01),
        BaseNoise(1, 40*args.training_ep//2, max_=0.3, min_=0.01)
    ]
    # steer_noise = OUNoise(1, sigma=0.5)
    # acc_noise = OUNoise(2, sigma=0.25)
    agent = Agent(
        args.gamma,
        args.tau,
        args.image_stack,
        args.buffer_capacity,
        args.batch_size,
        device=device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        critic_weight_decay=args.critic_wd,
        action_dim=action_dim,
        nb_updates=args.nb_updates,
    )
    env = Env(
        img_stack=args.image_stack,
        seed=args.train_seed,
        action_repeat=args.action_repeat,
        noise=add_noise,
        green_reward=args.green_reward,
        done_reward=args.done_reward,
    )
    eval_env = Env(
        img_stack=args.image_stack,
        seed=args.eval_seed,
        path_render=f"{run_name}" if args.eval_render else None,
        validations=args.eval_episodes,
        evaluation=True,
        action_repeat=args.action_repeat,
        # green_reward=args.green_reward,
        # done_reward=args.done_reward,
    )
    init_epoch = 0
    if args.from_checkpoint:
        init_epoch = agent.load_checkpoint(args.from_checkpoint)
    print(colored("Agent and environments created successfully", "green"))

    # Wandb config specification
    config = wandb.config
    config.args = args

    wandb.watch(agent.actor)

    noise_print = "not using noise"
    if env.use_noise:
        if env.generate_noise:
            noise_print = (
                f"using noise with [{env.noise_lower}, {env.noise_upper}] std bounds"
            )
        else:
            noise_print = f"using noise with [{env.random_noise}] std"

    print(
        colored(
            f"Training ddpg during {args.training_ep} epochs and {noise_print}",
            "magenta",
        )
    )

    trainer = Trainer(
        env,
        eval_env,
        agent,
        noises,
        args.training_ep,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        update_every=args.update_every,
        skip_zoom=args.skip_zoom,
        model_name=run_name
    )

    trainer.run()
