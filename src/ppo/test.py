import argparse
import torch
import os
import wandb
import glob
from termcolor import colored
from pyvirtualdisplay import Display
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('..')
from shared.utils.utils import init_uncert_file
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
    env_config.add_argument(
        "-N",
        "--noise",
        type=str,
        default="0,0.1",
        # default=None,
        help='Whether to use noise or not, and standard deviation bounds separated by comma (ex. "0,0.5")',
    )
    env_config.add_argument(
        "-NS", "--noise-steps", type=int, default=50, help="Number of noise steps",
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
        default=5,
        help="Number validations each noise step",
    )
    eval_config.add_argument(
        "-VR",
        "--val-render",
        action="store_true",
        help="render the environment on evaluation",
    )

    args = parser.parse_args()

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
    wandb.init(project="carracing-ppo")

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = Agent(args.nb_nets, args.img_stack, 0, model=args.model, device=device)
    env = None
    eval_env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.eval_seed,
        path_render=f"{args.model}" if args.val_render else None,
        validations=args.validations,
        evaluation=True,
        noise=add_noise,
    )
    agent.load(args.from_checkpoint, eval_mode=True)
    print(colored("Agent and environments created successfully", "green"))

    # Wandb config specification
    config = wandb.config
    config.learning_rate = agent.lr
    config.batch_size = agent.batch_size
    config.max_grad_norm = agent.max_grad_norm
    config.clip_param = agent.clip_param
    config.ppo_epoch = agent.ppo_epoch
    config.buffer_capacity = agent.buffer_capacity
    config.device = agent.device
    config.args = vars(args)

    if isinstance(agent._model._model, list):
        wandb.watch(agent._model._model[0])
    else:
        wandb.watch(agent._model._model)

    noise_print = "not using noise"
    if eval_env.use_noise:
        if eval_env.generate_noise:
            noise_print = f"using noise with [{eval_env.noise_lower}, {eval_env.noise_upper}] std bounds"
        else:
            noise_print = f"using noise with [{eval_env.random_noise}] std"

    print(
        colored(
            f"Testing {args.model} during {args.noise_steps} noise steps and {noise_print}",
            "magenta",
        )
    )

    for idx, noise in enumerate(tqdm(np.linspace(add_noise[0], add_noise[1], args.noise_steps))):
        eval_env.set_noise_value(noise)
        trainer = Trainer(
            agent,
            env,
            eval_env,
            0,
            nb_evaluations=args.validations,
            model_name=args.model,
        )
        trainer.eval(idx, mode="test")

