import argparse
import torch
import os
import wandb
import json
import glob
from termcolor import colored
from pyvirtualdisplay import Display

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
        "-N",
        "--noise",
        type=str,
        default="0,0.1",
        # default=None,
        help='Whether to use noise or not, and standard deviation bounds separated by comma (ex. "0,0.5")',
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
    agent_config.add_argument(
        '-FC',
        '--from-checkpoint', 
        type=str, 
        default=None, 
        help='Path to trained model')

    # Training Config
    train_config = parser.add_argument_group("Train config")
    train_config.add_argument(
        "-SZ",
        "--skip-zoom",
        type=int,
        default=0,
        help="Number of steps to skip at episode start",
    )
    train_config.add_argument(
        "-E", "--epochs", type=int, default=2000, help="Number of epochs"
    )
    train_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    train_config.add_argument(
        "-VI",
        "--val-interval",
        type=int,
        default=20,
        help="Interval between evaluations",
    )
    train_config.add_argument(
        "-V",
        "--validations",
        type=int,
        default=3,
        help="Number validations each 10 epochs",
    )
    train_config.add_argument(
        "-VR",
        "--val-render",
        action="store_true",
        help="render the environment on evaluation",
    )

    args = parser.parse_args()

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    if not os.path.exists("param"):
        os.makedirs("param")
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
    if not os.path.exists("uncertainties/train"):
        os.makedirs("uncertainties/train")
    init_uncert_file(file=f"uncertainties/train/{args.model}.txt")
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
    with open("config.json") as config_file:
        config = json.load(config_file)
    wandb.init(project=config["project"], entity=config["entity"])

    # print("Training model: {} with {} networks".format(args.model, args.uncert_q))

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = Agent(
        args.nb_nets, args.img_stack, args.gamma, model=args.model, device=device
    )
    env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.train_seed,
        noise=add_noise,
    )
    eval_env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.eval_seed,
        path_render=f"{args.model}" if args.val_render else None,
        validations=args.validations,
        evaluation=True,
        noise=add_noise,
    )
    init_epoch = 0
    if args.from_checkpoint:
        init_epoch = agent.load(args.from_checkpoint)
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
    config.args = args

    if isinstance(agent._model._model, list):
        wandb.watch(agent._model._model[0])
    else:
        wandb.watch(agent._model._model)

    noise_print = "not using noise"
    if env.use_noise:
        if env.generate_noise:
            noise_print = f"using noise with [{env.noise_lower}, {env.noise_upper}] std bounds"
        else:
            noise_print = f"using noise with [{env.random_noise}] std"

    print(
        colored(
            f"Training {args.model} during {args.epochs} epochs and {noise_print}",
            "magenta",
        )
    )

    trainer = Trainer(
        agent,
        env,
        eval_env,
        args.epochs,
        init_ep=init_epoch,
        nb_evaluations=args.validations,
        eval_interval=args.val_interval,
        model_name=args.model,
        skip_zoom=args.skip_zoom,
    )

    trainer.run()
