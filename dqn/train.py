import argparse
import numpy as np
import torch
import wandb
import json
import os
import glob
from termcolor import colored

from pyvirtualdisplay import Display

from components import make_agent, Env, Trainer
from utilities import init_uncert_file


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
        default="0,0.3",
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
        "-T",
        "--tau",
        type=float,
        default=0.1,
        help="DDQN Tau parameter, only used when model is ddqn2015",
    )
    agent_config.add_argument(
        "-NTR",
        "--nb-target-replace",
        type=int,
        default=5,
        help="Number episodes target network replace, only used when model is dqn",
    )
    agent_config.add_argument(
        "-LR", "--learning-rate", type=float, default=0.001, help="Learning Rate"
    )
    agent_config.add_argument(
        "-G", "--gamma", type=float, default=0.95, help="Discount factor"
    )
    agent_config.add_argument(
        "-IS", "--image-stack", type=int, default=4, help="Number of images to stack"
    )
    agent_config.add_argument(
        "-BC",
        "--buffer-capacity",
        type=int,
        default=8000,
        help="Replay buffer capacity",
    )
    agent_config.add_argument(
        "-BS", "--batch-size", type=int, default=16, help="Batch size"
    )
    agent_config.add_argument(
        "-FC", "--from-checkpoint", type=str, default=None, help="Path to trained model"
    )

    # Epsilon Config
    epsilon_config = parser.add_argument_group("Epsilon config")
    epsilon_config.add_argument(
        "-EM",
        "--epsilon-method",
        type=str,
        default="exp",
        help="Epsilon decay method: constant, linear, exp or inverse_sigmoid",
    )
    epsilon_config.add_argument(
        "-EMa",
        "--epsilon-max",
        type=float,
        default=1,
        help="The minimum value of epsilon, used this value in constant",
    )
    epsilon_config.add_argument(
        "-EMi",
        "--epsilon-min",
        type=float,
        default=0.1,
        help="The minimum value of epsilon",
    )
    epsilon_config.add_argument(
        "-EF",
        "--epsilon-factor",
        type=float,
        default=6,
        help="Factor parameter of epsilon decay, only used when method is exp or inverse_sigmoid",
    )
    epsilon_config.add_argument(
        "-EMS",
        "--epsilon-max-steps",
        type=int,
        default=90000,
        help="Max Epsilon Steps parameter, when epsilon is close to the minimum",
    )

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
        "-NTS", "--training-ep", type=int, default=2000, help="Number traning episodes"
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
        "-ER",
        "--eval-render",
        action="store_true",
        help="Whether to render evaluation or not",
    )
    train_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    args = parser.parse_args()

    old_settings = np.seterr(all="raise")

    do_nothing_action = tuple([[0, 0, 0]])
    full_actions = (
        [-1, 0, 0],  # Turn Left
        [1, 0, 0],  # Turn Right
        [0, 0, 1],  # Full Break
        [0, 1, 0],  # Accelerate
        [-1, 1, 0],  # Left accelerate
        [1, 1, 0],  # Right accelerate
        [-1, 0, 1],  # Left break
        [1, 0, 1],  # Right break
    )
    actions = (
        do_nothing_action
        + full_actions
        + make_soft_actions(full_actions, 0.25)
        + make_soft_actions(full_actions, 0.5)
        + make_soft_actions(full_actions, 0.75)
    )

    # actions = (
    #     [-1, 0, 0],  # Turn Left
    #     [1, 0, 0],  # Turn Right
    #     [0, 0, 1],  # Full Break
    #     [0, 1, 0],  # Accelerate
    #     [0, 0, 0],  # Do nothing
    #     [-1, 1, 0],  # Left accelerate
    #     [1, 1, 0],  # Right accelerate
    #     [-1, 0, 1],  # Left break
    #     [1, 0, 1],  # Right break
    #     [-0.5, 0, 0],  # Soft left
    #     [0.5, 0, 0],  # Soft right
    #     [0, 0, 0.5],  # Soft break
    #     [0, 0.5, 0],  # Soft accelerate
    #     [-0.5, 0.5, 0],  # Soft Left accelerate
    #     [0.5, 0.5, 0],  # Soft Right accelerate
    #     [-0.5, 0, 0.5],  # Soft Left break
    #     [0.5, 0, 0.5],  # Soft Right break
    # )

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

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = make_agent(
        args.model,
        args.image_stack,
        actions,
        args.learning_rate,
        args.gamma,
        args.buffer_capacity,
        args.batch_size,
        device=device,
        clip_grad=False,
        epsilon_method=args.epsilon_method,
        epsilon_max=args.epsilon_max,
        epsilon_min=args.epsilon_min,
        epsilon_factor=args.epsilon_factor,
        epsilon_max_steps=args.epsilon_max_steps,
        tau=args.tau,
        nb_target_replace=args.nb_target_replace,
    )
    env = Env(
        img_stack=args.image_stack,
        seed=args.train_seed,
        action_repeat=args.action_repeat,
        noise=add_noise,
    )
    eval_env = Env(
        img_stack=args.image_stack,
        seed=args.eval_seed,
        path_render=f"{args.model}" if args.eval_render else None,
        validations=args.eval_episodes,
        evaluation=True,
        action_repeat=args.action_repeat,
        noise=add_noise,
    )
    init_epoch = 0
    if args.from_checkpoint:
        init_epoch = agent.load(args.from_checkpoint)
    print(colored("Agent and environments created successfully", "green"))

    # Wandb config specification
    config = wandb.config
    config.args = args

    if args.model.lower() in ["dqn", "ddqn2015"]:
        wandb.watch(agent._model)
    elif args.model.lower() in ["ddqn2018"]:
        wandb.watch(agent._model1)

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
            f"Training {args.model} during {args.training_ep} epochs and {noise_print}",
            "magenta",
        )
    )

    trainer = Trainer(
        env,
        eval_env,
        agent,
        args.training_ep,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        skip_zoom=args.skip_zoom,
        model_name=args.model,
    )

    trainer.run()
