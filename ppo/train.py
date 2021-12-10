import argparse
import numpy as np
import torch
import os
import wandb
import json
import glob
from tqdm import tqdm
from termcolor import colored

from utilities import save_uncert, init_uncert_file
from components import Agent, Env

from pyvirtualdisplay import Display


class Trainer:
    def __init__(
        self,
        agent: Agent,
        env: Env,
        eval_env: Env,
        episodes: int,
        init_ep: int = 0,
        nb_evaluations: int = 1,
        eval_interval: int = 10,
        skip_zoom=None,
        model_name="base",
        checkpoint_every=10,
    ) -> None:
        self._agent = agent
        self._env = env
        self._eval_env = eval_env
        self._init_ep = init_ep
        self._nb_episodes = episodes
        self._nb_evaluations = nb_evaluations
        self._eval_interval = eval_interval
        self._skip_zoom = skip_zoom
        self._model_name = model_name
        self._checkpoint_every = checkpoint_every

        self._best_score = -100
        self._eval_nb = 0

    def run(self):
        running_score = 0

        for i_ep in tqdm(range(self._init_ep, self._nb_episodes)):
            score = 0
            steps = 0
            state = self._env.reset()

            if self._skip_zoom is not None:
                for _ in range(self._skip_zoom):
                    state, _, _, _ = self._env.step([0, 0, 0])

            for _ in range(1000):
                action, a_logp = self._agent.select_action(state)[:2]
                state_, reward, done, die = self._env.step(
                    action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
                )
                if self._agent.store_transition(
                    (state, action, a_logp, reward, state_)
                ):
                    print("updating")
                    self._agent.update()
                score += reward
                state = state_
                steps += 1

                if done or die:
                    break
            running_score = running_score * 0.99 + score * 0.01
            wandb.log(
                {
                    "Train Episode": i_ep,
                    "Episode Running Score": float(running_score),
                    "Episode Score": float(score),
                    "Episode Steps": float(steps),
                }
            )

            if (i_ep + 1) % self._eval_interval == 0:
                eval_score = self.eval(i_ep)

                if eval_score > self._best_score:
                    self._agent.save(i_ep, path=f"param/best_{self._model_name}.pkl")
                    self._best_score = eval_score

            if (i_ep + 1) % self._checkpoint_every == 0:
                self._agent.save(i_ep, path=f"param/checkpoint_{self._model_name}.pkl")

            if running_score > self._env.reward_threshold:
                print(
                    "Solved! Running reward is now {} and the last episode runs to {}!".format(
                        running_score, score
                    )
                )
                self._agent.save(i_ep, path=f"param/best_{self._model_name}.pkl")
                break

    def eval(self, episode_nb):
        # agent.eval_mode()
        mean_score = 0
        mean_uncert = np.array([0, 0], dtype=np.float64)
        mean_steps = 0

        for i_val in range(self._nb_evaluations):

            score = 0
            steps = 0
            state = self._env.reset()
            die = False

            uncert = []
            while not die:
                action, _, (epis, aleat) = agent.select_action(state, eval=True)
                uncert.append(
                    [epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]]
                )
                state_, reward, _, die = env.step(
                    action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
                )
                score += reward
                state = state_
                steps += 1

            uncert = np.array(uncert)
            save_uncert(
                episode_nb,
                i_val,
                score,
                uncert,
                file=f"uncertainties/train/train_{self._model_name}.txt",
            )

            mean_uncert += np.mean(uncert, axis=0) / self._nb_evaluations
            mean_score += score / self._nb_evaluations
            mean_steps += steps / self._nb_evaluations

        wandb.log(
            {
                "Eval Episode": self._eval_nb,
                "Eval Mean Score": float(mean_score),
                "Eval Mean Epist Uncert": float(mean_uncert[0]),
                "Eval Mean Aleat Uncert": float(mean_uncert[1]),
                "Eval Mean Steps": float(mean_steps),
            }
        )

        self._eval_nb += 1
        print(
            "Eval score: {}\tSteps: {}\tUncertainties: {}".format(
                mean_score, mean_steps, mean_uncert
            )
        )
        # agent.train_mode()

        return mean_score


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
        default="0,0.3",
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
        "-FC",
        "--from-checkpoint",
        action="store_true",
        help="Whether to use checkpoint file",
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
    init_uncert_file(file=f"uncertainties/train/train_{args.model}.txt")
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
        print_arg = f"noisy observation with {args.noise} std bounds"
    else:
        add_noise = None
        print_arg = ""

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
        init_epoch = agent.load()
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

    print(
        colored(
            f"Training {args.model} during {args.epochs} epochs and {print_arg}",
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
