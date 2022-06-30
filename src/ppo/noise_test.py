import argparse
import torch
from termcolor import colored
from pyvirtualdisplay import Display

import sys
sys.path.append('..')
from shared.components.evaluator import Evaluator
from components.agent import Agent

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

    # Agent Config
    agent_config = parser.add_argument_group("Agent config")
    agent_config.add_argument(
        "-M",
        "--model",
        type=str,
        default="base",
        help='Type of uncertainty model: "base", "sensitivity", "dropout", "bootstrap", "aleatoric", "bnn" or "custom"',
    )
    agent_config.add_argument(
        "-NN",
        "--nb-nets",
        type=int,
        default=10,
        help="Number of networks to estimate uncertainties",
    )
    agent_config.add_argument(
        "-G", "--gamma", type=float, default=0.99, help="discount factor"
    )
    agent_config.add_argument(
        "-IS", "--img-stack", type=int, default=4, help="stack N images in a state"
    )
    # Training Config
    train_config = parser.add_argument_group("Train config")
    train_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    args = parser.parse_args()

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()
    
    seed = 0

    # Whether to use cuda or cpu
    if args.device == "auto":
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
    else:
        device = args.device
    print(colored(f"Using: {device}", "green"))

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = Agent(
        args.nb_nets,
        args.img_stack,
        0,
        model=args.model,
        device=device
    )
    agent.load(f"param/best_{args.model}.pkl", eval_mode=True)
    print(colored("Agent and environments created successfully", "green"))

    evaluator = Evaluator(
        args.img_stack,
        args.action_repeat,
        args.model,
        device=device,
        base_path='uncertainties/noisetest',
        nb=3,
    )
    evaluator.noise_eval(0, agent)

    print(colored("\nTest completed", "green"))