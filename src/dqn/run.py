import argparse
import numpy as np
import torch
import wandb
import os
import glob
from termcolor import colored
from pyvirtualdisplay import Display
import uuid
from collections import namedtuple
from tqdm import tqdm

import sys
sys.path.append('..')
from utilities.eps_scheduler import Epsilon
from utilities.replay_buffer import ReplayMemory
from components.uncert_agents import make_agent
from components.trainer import Trainer
from shared.components.evaluator import Evaluator
from shared.components.env import Env
from shared.utils.utils import init_uncert_file


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
        "-TSE",
        "--test-seed",
        type=float,
        default=20,
        help="Testing Environment Random seed",
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
        default=1000000,
        help="Replay buffer capacity",
    )
    agent_config.add_argument(
        "-BS", "--batch-size", type=int, default=64, help="Batch size"
    )
    agent_config.add_argument(
        "-A", "--actions", type=str, default="0.25,0.5", help="Basic actions multipliers as list, for example '0.25,0.5'"
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
        default="linear",
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
        default=0.05,
        help="The minimum value of epsilon",
    )
    epsilon_config.add_argument(
        "-EF",
        "--epsilon-factor",
        type=float,
        default=7,
        help="Factor parameter of epsilon decay, only used when method is exp or inverse_sigmoid",
    )
    epsilon_config.add_argument(
        "-EMS",
        "--epsilon-max-steps",
        type=int,
        default=1700,
        help="Max Epsilon Steps parameter, when epsilon is close to the minimum",
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
        "-NTE", "--training-ep", type=int, default=3400, help="Number traning episodes"
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
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )

    # Test Config
    test_config = parser.add_argument_group("Test config")
    test_config.add_argument(
        "-TEp",
        "--test-episodes",
        type=int,
        default=3,
        help="Number of testing episodes",
    )
    test_config.add_argument(
        "-TR",
        "--test-render",
        action="store_true",
        help="Whether to render testing or not",
    )
    test_config.add_argument(
        "-OT",
        "--ommit-training",
        action="store_true",
        help="Whether to ommit training the agent or not",
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

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    # Create param directory
    if not os.path.exists("param"):
        os.makedirs("param")
    # Create uncertainties directory
    if not os.path.exists("uncertainties"):
        os.makedirs("uncertainties")
    if not os.path.exists("uncertainties/eval"):
        os.makedirs("uncertainties/eval")
    if not os.path.exists("uncertainties/test"):
        os.makedirs("uncertainties/test")
    if not os.path.exists("uncertainties/test0"):
        os.makedirs("uncertainties/test0")

    # Create render folders
    if not os.path.exists("render"):
        os.makedirs("render")
    if not os.path.exists("render/train"):
        os.makedirs("render/train")
    if not os.path.exists("render/test"):
        os.makedirs("render/test")
    if not os.path.exists(f"render/train/{run_name}"):
        os.makedirs(f"render/train/{run_name}")
    elif not args.ommit_training:
        files = glob.glob(f"render/train/{run_name}/*")
        for f in files:
            os.remove(f)
    if not os.path.exists(f"render/test/{run_name}"):
        os.makedirs(f"render/test/{run_name}")
    else:
        files = glob.glob(f"render/test/{run_name}/*")
        for f in files:
            os.remove(f)
    
    if not args.ommit_training:
        init_uncert_file(file=f"uncertainties/eval/{run_name}.txt")
    init_uncert_file(file=f"uncertainties/test/{run_name}.txt")
    init_uncert_file(file=f"uncertainties/test0/{run_name}.txt")
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
    wandb.init(project="carracing-dqn")

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    Transition = namedtuple(
        "Transition", ("state", "action", "next_state", "reward", "done")
    )
    buffer = ReplayMemory(
        args.buffer_capacity,
        args.batch_size,
        Transition,
    )
    epsilon = Epsilon(
        args.epsilon_max_steps,
        method=args.epsilon_method,
        epsilon_max=args.epsilon_max,
        epsilon_min=args.epsilon_min,
        factor=args.epsilon_factor,
    )
    agent = make_agent(
        args.model,
        args.nb_nets,
        args.image_stack,
        actions,
        args.learning_rate,
        args.gamma,
        buffer,
        epsilon,
        args.batch_size,
        device=device,
        clip_grad=False,
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
        path_render=f"./render/train/{run_name}" if args.eval_render else None,
        validations=args.eval_episodes,
        evaluation=True,
        action_repeat=args.action_repeat,
        # green_reward=args.green_reward,
        # done_reward=args.done_reward,
    )
    evaluator = None
    if args.model != 'base' and not args.ommit_training:
        evaluator = Evaluator(
            args.image_stack,
            args.action_repeat,
            args.model,
            device=device,
        )
    init_epoch = 0
    if args.from_checkpoint:
        init_epoch = agent.load_param(args.from_checkpoint)
    print(colored("Agent and environments created successfully", "green"))

    # Wandb config specification
    config = wandb.config
    config.args = vars(args)

    # wandb.watch(agent._model1)

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
            f"Training ddqn {type(agent)} during {args.training_ep} epochs and {noise_print}",
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
        model_name=run_name,
        evaluator=evaluator,
    )

    if not args.ommit_training:
        trainer.run()
    else:
        print(colored("\nTraining Ommited", "magenta"))
    env.close()
    eval_env.close()
    del env
    del eval_env
    del trainer
    del agent
    del evaluator

    print(colored("\nTraining completed, now testing", "green"))
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
    test_env = Env(
        img_stack=args.image_stack,
        seed=args.test_seed,
        path_render=f"./render/test/{args.model}" if args.test_render else None,
        validations=args.test_episodes,
        evaluation=True,
        action_repeat=args.action_repeat,
        noise=add_noise,
    )
    # evaluator = Evaluator(
    #     args.image_stack,
    #     args.action_repeat,
    #     args.model,
    #     device=device,
    #     base_path='uncertainties/customtest'
    # )
    agent.load_param(f"param/best_{args.model}.pkl", eval_mode=True)
    print(colored("Agent and environments created successfully", "green"))

    noise_print = "not using noise"
    if test_env.use_noise:
        if test_env.generate_noise:
            noise_print = f"using noise with [{test_env.noise_lower}, {test_env.noise_upper}] std bounds"
        else:
            noise_print = f"using noise with [{test_env.random_noise}] std"

    print(
        colored(
            f"Testing {args.model} during {args.noise_steps} noise steps and {noise_print}",
            "magenta",
        )
    )

    # Test increasing noise
    for idx, noise in enumerate(tqdm(np.linspace(add_noise[0], add_noise[1], args.noise_steps))):
        test_env.set_noise_value(noise)
        # if evaluator:
        #     evaluator.set_noise_value(noise)
        trainer = Trainer(
            None,
            test_env,
            agent,
            0,
            eval_episodes=args.test_episodes,
            model_name=args.model,
            # evaluator=evaluator,
        )
        trainer.eval(idx, mode="test")
    
    # Test noise 0
    # evaluator = Evaluator(
    #     args.image_stack,
    #     args.action_repeat,
    #     args.model,
    #     device=device,
    #     base_path='uncertainties/customtest0'
    # )
    test_env.use_noise = False
    for idx in tqdm(range(args.noise_steps)):
        trainer = Trainer(
            None,
            test_env,
            agent,
            0,
            eval_episodes=args.test_episodes,
            model_name=args.model,
            # evaluator=evaluator,
        )
        trainer.eval(idx, mode="test0")

    # Test controller 1 and 2
    evaluator = Evaluator(
        args.image_stack,
        args.action_repeat,
        args.model,
        device=device,
        base_path='uncertainties/customtest1'
    )
    evaluator.eval(0, agent)

    evaluator = Evaluator(
        args.image_stack,
        args.action_repeat,
        args.model,
        device=device,
        base_path='uncertainties/customtest2',
        nb=2,
    )
    evaluator.eval2(0, agent)

    print(colored("\nTest completed", "green"))