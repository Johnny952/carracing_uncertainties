import numpy as np
import wandb
from tqdm import tqdm

from shared.utils.utils import save_uncert
from shared.components.env import Env
from components.agent import Agent
from shared.components.evaluator import Evaluator

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
        skip_zoom = None,
        model_name = "base",
        checkpoint_every = 10,
        debug=False,
        evaluator: Evaluator = None,
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
        self._debug = debug
        self._evaluator = evaluator

        self._best_score = -100
        self._eval_nb = 0

    def run(self):
        running_score = 0

        for i_ep in tqdm(range(self._init_ep, self._nb_episodes), 'Training'):
            score = 0
            steps = 0
            state = self._env.reset()

            if self._skip_zoom is not None:
                for _ in range(self._skip_zoom):
                    state = self._env.step([0, 0, 0])[0]

            for _ in range(1000):
                action, a_logp = self._agent.select_action(state)[:2]
                state_, reward, done, die = self._env.step(
                    action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
                )[:4]
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

                if eval_score > self._best_score and not self._debug:
                    self._agent.save(i_ep, path=f"param/best_{self._model_name}.pkl")
                    self._best_score = eval_score

            if (i_ep + 1) % self._checkpoint_every == 0 and not self._debug:
                self._agent.save(i_ep, path=f"param/checkpoint_{self._model_name}.pkl")

            if running_score > self._env.reward_threshold:
                print(
                    "Solved! Running reward is now {} and the last episode runs to {}!".format(
                        running_score, score
                    )
                )
                if not self._debug:
                    self._agent.save(i_ep, path=f"param/best_{self._model_name}.pkl")
                break

    def eval(self, episode_nb, mode='eval'):
        if self._evaluator:
            self._evaluator.eval(episode_nb, self._agent)
        return self.base_eval(episode_nb, mode=mode)

    def base_eval(self, episode_nb, mode='train'):
        assert mode in ['train', 'eval', 'test0', 'test']
        # self._agent.eval_mode()
        mean_score = 0
        mean_uncert = np.array([0, 0], dtype=np.float64)
        mean_steps = 0

        for i_val in tqdm(range(self._nb_evaluations), f'{mode.title()} ep {episode_nb}'):

            score = 0
            steps = 0
            state = self._eval_env.reset()
            die = False

            uncert = []
            while not die:
                action, _, (epis, aleat) = self._agent.select_action(state, eval=True)
                uncert.append(
                    [epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]]
                )
                state_, reward, _, die = self._eval_env.step(
                    action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
                )[:4]
                score += reward
                state = state_
                steps += 1

            uncert = np.array(uncert)
            if not self._debug:
                save_uncert(
                    episode_nb,
                    i_val,
                    score,
                    uncert,
                    file=f"uncertainties/{mode}/{self._model_name}.txt",
                    sigma=self._eval_env.random_noise,
                )

            mean_uncert += np.mean(uncert, axis=0) / self._nb_evaluations
            mean_score += score / self._nb_evaluations
            mean_steps += steps / self._nb_evaluations

        wandb_mode = mode.title()
        wandb.log(
            {
                f"{wandb_mode} Episode": self._eval_nb,
                f"{wandb_mode} Mean Score": float(mean_score),
                f"{wandb_mode} Mean Epist Uncert": float(mean_uncert[0]),
                f"{wandb_mode} Mean Aleat Uncert": float(mean_uncert[1]),
                f"{wandb_mode} Mean Steps": float(mean_steps),
            }
        )

        self._eval_nb += 1

        return mean_score