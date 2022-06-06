import argparse
import torch
import os
import glob
from termcolor import colored
from pyvirtualdisplay import Display

import sys
sys.path.append('../..')
sys.path.append('..')
from shared.components.env import Env
from components.agent import Agent
from components.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for the CarRacing-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment Config
    env_config = parser.add_argument_group("Environment config")
    env_config.add_argument(
        "-AR", "--action-repeat", type=int, default=8, help="repeat action in N frames"
    )
    env_config.add_argument(
        "-ES",
        "--eval-seed",
        type=float,
        default=10,
        help="Evaluation Environment Random seed",
    )

    # Agent Config
    agent_config = parser.add_argument_group("Agent config")
    agent_config.add_argument(
        "-IS", "--img-stack", type=int, default=4, help="stack N images in a state"
    )
    agent_config.add_argument(
        "-FC",
        "--from-checkpoint",
        type=str,
        required=True,
        # default='param/ppo_net_params_base.pkl',
        help="Path to trained model",
    )

    # Eval Config
    eval_config = parser.add_argument_group("Evaluation config")
    eval_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    eval_config.add_argument(
        "-V",
        "--validations",
        type=int,
        default=1,
        help="Number validations",
    )
    eval_config.add_argument(
        "-VR",
        "--val-render",
        action="store_true",
        help="render the environment on evaluation",
    )
    eval_config.add_argument(
        "-DB",
        "--debug",
        action="store_true",
        help="debug mode",
    )

    args = parser.parse_args()

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    if not args.debug:
        if not os.path.exists("images"):
            os.makedirs("images")
        if not os.path.exists("render") and args.val_render:
            os.makedirs("render")
        else:
            files = glob.glob(f"render/*")
            for f in files:
                os.remove(f)
    print(colored("Data folders created successfully", "green"))

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Whether to use cuda or cpu
    if args.device == "auto":
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.eval_seed)
        if use_cuda:
            torch.cuda.manual_seed(args.eval_seed)
    else:
        device = args.device
    print(colored(f"Using: {device}", "green"))

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = Agent(
        0,
        args.img_stack,
        0,
        model='vae',
        device=device
    )
    env = None
    eval_env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.eval_seed,
        path_render=f"render" if args.val_render else None,
        validations=args.validations,
        evaluation=True,
    )
    agent.load(args.from_checkpoint, eval_mode=True)
    print(colored("Agent and environments created successfully", "green"))

    trainer = Trainer(
        agent,
        env,
        eval_env,
        0,
        nb_evaluations=args.validations,
        model_name='vae',
        debug=args.debug,
    )
    trainer.vae_eval(directory='images', device=device)