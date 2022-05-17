import argparse
import torch
import os
import wandb
import glob
from termcolor import colored
from pyvirtualdisplay import Display
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('..')
from shared.utils.utils import init_uncert_file
from shared.components.env import Env
from shared.components.evaluator import Evaluator
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
        "-TSE",
        "--test-seed",
        type=float,
        default=20,
        help="Test Environment Random seed",
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
        "-E", "--epochs", type=int, default=3000, help="Number of epochs"
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
    train_config.add_argument(
        "-DB",
        "--debug",
        action="store_true",
        help="debug mode",
    )

    # Test Config
    test_config = parser.add_argument_group("Test config")
    test_config.add_argument(
        "-TE",
        "--test-episodes",
        type=int,
        default=5,
        help="Number validations each noise step",
    )
    test_config.add_argument(
        "-TR",
        "--test-render",
        action="store_true",
        help="render the environment on testing",
    )
    test_config.add_argument(
        "-OT",
        "--ommit-traning",
        action="store_true",
        help="Whether to ommit training the agent or not",
    )

    args = parser.parse_args()

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    if not args.debug:
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
        if not os.path.exists(f"render/train/{args.model}"):
            os.makedirs(f"render/train/{args.model}")
        elif not args.ommit_training:
            files = glob.glob(f"render/train/{args.model}/*")
            for f in files:
                os.remove(f)
        if not os.path.exists(f"render/test/{args.model}"):
            os.makedirs(f"render/test/{args.model}")
        else:
            files = glob.glob(f"render/test/{args.model}/*")
            for f in files:
                os.remove(f)
        
        if not args.ommit_training:
            init_uncert_file(file=f"uncertainties/eval/{args.model}.txt")
        init_uncert_file(file=f"uncertainties/test/{args.model}.txt")
        init_uncert_file(file=f"uncertainties/test0/{args.model}.txt")
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
    wandb.init(project="carracing-ppo")

    # Noise parser
    if args.noise:
        add_noise = [float(bound) for bound in args.noise.split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = Agent(
        args.nb_nets,
        args.img_stack,
        args.gamma,
        model=args.model,
        device=device,
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
        path_render=f"./render/train/{args.model}" if args.val_render else None,
        validations=args.validations,
        evaluation=True,
        # noise=add_noise,
    )
    evaluator = None
    if args.model != 'base':
        evaluator = Evaluator(
            args.img_stack,
            args.action_repeat,
            args.model,
            device=device,
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
    config.args = vars(args)

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
        debug=args.debug,
        evaluator=evaluator,
    )

    if not args.ommit_training:
        print(colored("\nTraining Ommited", "magenta"))
        trainer.run()
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
    agent = Agent(
        args.nb_nets,
        args.img_stack,
        0,
        model=args.model,
        device=device
    )
    test_env = Env(
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
        seed=args.test_seed,
        path_render=f"./render/test/{args.model}" if args.test_render else None,
        validations=args.test_episodes,
        evaluation=True,
        noise=add_noise,
    )
    evaluator = None
    if args.model != 'base':
        evaluator = Evaluator(
            args.img_stack,
            args.action_repeat,
            args.model,
            device=device,
            base_path='uncertainties/customtest'
        )
    agent.load(f"param/best_{args.model}.pkl", eval_mode=True)
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

    for idx, noise in enumerate(tqdm(np.linspace(add_noise[0], add_noise[1], args.noise_steps))):
        test_env.set_noise_value(noise)
        if evaluator:
            evaluator.set_noise_value(noise)
        trainer = Trainer(
            agent,
            None,
            test_env,
            0,
            nb_evaluations=args.test_episodes,
            model_name=args.model,
            debug=args.debug,
            evaluator=evaluator,
        )
        trainer.eval(idx, mode="test")
    

    evaluator = None
    if args.model != 'base':
        evaluator = Evaluator(
            args.image_stack,
            args.action_repeat,
            args.model,
            device=device,
            base_path='uncertainties/customtest0'
        )
    test_env.use_noise = False
    trainer = Trainer(
        agent,
        None,
        test_env,
        0,
        nb_evaluations=args.noise_steps,
        model_name=args.model,
        debug=args.debug,
        evaluator=evaluator,
    )
    trainer.eval(idx, mode="test0")
    print(colored("\nTest completed", "green"))
