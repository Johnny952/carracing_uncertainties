import argparse
import numpy as np
import torch
import wandb
import os
import glob
from termcolor import colored
from tqdm import tqdm
from pyvirtualdisplay import Display

import sys
sys.path.append('..')
from shared.components.env import Env
from shared.utils.utils import init_uncert_file
from components.uncert_agents import make_agent
from components.trainer import Trainer

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
        "-N",
        "--noise",
        type=str,
        default="0,0.1",
        # default=None,
        help='Whether to use noise or not, and standard deviation bounds separated by comma (ex. "0,0.5")',
    )
    env_config.add_argument(
        "-NS", "--noise-steps", type=int, default=25, help="Number of noise steps",
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
        "-M",
        "--model",
        type=str,
        default="ddqn2018",
        help="Which RL model use: dqn, ddqn2015 or ddqn2018",
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
    agent_config.add_argument(
        "-FC",
        "--from-checkpoint",
        type=str,
        required=True,
        help="Path to trained model",
    )

    # Eval Config
    test_config = parser.add_argument_group("Test config")
    test_config.add_argument(
        "-EEp",
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes",
    )
    test_config.add_argument(
        "-ER",
        "--eval-render",
        action="store_true",
        help="Whether to render evaluation or not",
    )
    test_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    args = parser.parse_args()

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

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    if not os.path.exists("uncertainties"):
        os.makedirs("uncertainties")
    if not os.path.exists("render"):
        os.makedirs("render")
    if not os.path.exists(f"render/{args.model}"):
        os.makedirs(f"render/{args.model}")
    else:
        files = glob.glob(f"render/{args.model}/*")
        for f in files:
            os.remove(f)
    if not os.path.exists("uncertainties/test"):
        os.makedirs("uncertainties/test")
    init_uncert_file(file=f"uncertainties/test/{args.model}.txt")
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

    # Init Wandb
    wandb.init(project="carracing-dqn")

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = make_agent(
        args.model,
        args.nb_nets,
        args.image_stack,
        actions,
        0.001,
        0.95,
        None,
        None,
        64,
        device=device,
        clip_grad=False,
    )
    env = None
    eval_env = Env(
        img_stack=args.image_stack,
        seed=args.eval_seed,
        path_render=f"{args.model}" if args.eval_render else None,
        validations=args.eval_episodes,
        evaluation=True,
        action_repeat=args.action_repeat,
        noise=add_noise,
    )
    agent.load(args.from_checkpoint)
    print(colored("Agent and environments created successfully", "green"))

    # Wandb config specification
    config = wandb.config
    config.args = vars(args)

    noise_print = "not using noise"
    if env.use_noise:
        if env.generate_noise:
            noise_print = f"using noise with [{env.noise_lower}, {env.noise_upper}] std bounds"
        else:
            noise_print = f"using noise with [{env.random_noise}] std"

    print(
        colored(
            f"Testing {args.model} during {args.noise_steps} noise steps and {noise_print}",
            "magenta",
        )
    )

    for noise in tqdm(np.linspace(add_noise[0], add_noise[1], args.noise_steps)):
        eval_env.set_noise_value(noise)
        trainer = Trainer(
            env,
            eval_env,
            agent,
            0,
            eval_episodes=args.eval_episodes,
            model_name=args.model,
        )
        trainer.eval(0, mode="test")

