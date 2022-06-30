import argparse
import numpy as np
import torch
from termcolor import colored
from pyvirtualdisplay import Display

import sys
sys.path.append('..')
from components.uncert_agents import make_agent
from shared.components.evaluator import Evaluator


STEER_RANGE = [-0.5, 0.5]
THROTTLE_RANGE = [0, 1]
BRAKE_RANGE = [0, 0.2]


def make_soft_actions(actions: list, factor: float):
    soft_actions = np.array(actions) * factor
    return tuple(soft_actions.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for the CarRacing-v0",
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
        "-M",
        "--model",
        type=str,
        default="base",
        help='Type of uncertainty model: "base", "sensitivity", "dropout", "bootstrap", "aleatoric", "bnn"',
    )
    agent_config.add_argument(
        "-NN",
        "--nb-nets",
        type=int,
        default=10,
        help="Number of networks to estimate uncertainties",
    )
    agent_config.add_argument(
        "-IS", "--image-stack", type=int, default=4, help="Number of images to stack"
    )
    agent_config.add_argument(
        "-A", "--actions", type=str, default="0.25,0.5", help="Basic actions multipliers as list, for example '0.25,0.5'"
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
    
    # run_name = f"{args.model}_{uuid.uuid4()}"
    run_name = args.model
    
    old_settings = np.seterr(all="raise")
    
    do_nothing_action = tuple([[0, THROTTLE_RANGE[0], BRAKE_RANGE[0]]])
    full_actions = (
        [STEER_RANGE[0], THROTTLE_RANGE[0], BRAKE_RANGE[0]],  # Turn Left
        [STEER_RANGE[1], THROTTLE_RANGE[0], BRAKE_RANGE[0]],  # Turn Right
        [0, THROTTLE_RANGE[0], BRAKE_RANGE[1]],  # Full Break
        [0, THROTTLE_RANGE[1], BRAKE_RANGE[0]],  # Accelerate
        [STEER_RANGE[0], THROTTLE_RANGE[1], BRAKE_RANGE[0]],  # Left accelerate
        [STEER_RANGE[1], THROTTLE_RANGE[1], BRAKE_RANGE[0]],  # Right accelerate
        [STEER_RANGE[0], THROTTLE_RANGE[0], BRAKE_RANGE[1]],  # Left break
        [STEER_RANGE[1], THROTTLE_RANGE[0], BRAKE_RANGE[1]],  # Right break
    )
    alter_actions = ()
    args_actions = [float(i.strip()) for i in args.actions.split(',')]
    for mult in args_actions:
        alter_actions += make_soft_actions(full_actions, mult)
    actions = (
        do_nothing_action
        + full_actions
        + alter_actions
    )

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Whether to use cuda or cpu
    seed = 0

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
    agent = make_agent(
        args.model,
        args.nb_nets,
        args.image_stack,
        actions,
        0,
        1,
        None,
        None,
        1,
        device=device,
    )
    agent.load_param(f"param/best_{args.model}.pkl", eval_mode=True)
    print(colored("Agent and environments created successfully", "green"))

    evaluator = Evaluator(
        args.image_stack,
        args.action_repeat,
        args.model,
        device=device,
        base_path='uncertainties/noisetest',
        nb=3,
    )
    evaluator.noise_eval(0, agent)

    print(colored("\nTest completed", "green"))