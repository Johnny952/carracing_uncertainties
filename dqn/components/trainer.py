import numpy as np
import wandb
from tqdm import tqdm

from components import Env
from components.agent import Agent
from utilities import save_uncert


class Trainer:
    def __init__(
        self,
        env: Env,
        eval_env: Env,
        agent: Agent,
        nb_training_ep: int,
        eval_episodes: int = 3,
        eval_every: int = 10,
        skip_zoom=None,
        model_name: str = "base",
        checkpoint_every: int = 10,
    ) -> None:
        self._env = env
        self._eval_env = eval_env
        self._agent = agent
        self._nb_training_ep = nb_training_ep
        self._eval_episodes = eval_episodes
        self._eval_every = eval_every
        self._skip_zoom = skip_zoom
        self._model_name = model_name
        self._checkpoint_every = checkpoint_every

        self._best_score = -100
        self._eval_nb = 0

    def run(self):
        running_score = 0

        for episode_nb in tqdm(range(self._nb_training_ep)):
            ob_t = self._env.reset()
            score = 0
            steps = 0

            if self._skip_zoom is not None:
                for _ in range(self._skip_zoom):
                    ob_t, _, _, _ = self._env.step([0, 0, 0])

            for _ in range(1000):
                action, action_idx = self._agent.select_action(ob_t)
                ob_t1, reward, done, die = self._env.step(action)
                if self._agent.store_transition(
                    ob_t, action_idx, ob_t1, reward, (done or die)
                ):
                    self._agent.update()

                score += reward
                ob_t = ob_t1
                steps += 1

                if done or die:
                    break

            running_score = running_score * 0.99 + score * 0.01
            wandb.log(
                {
                    "Train Episode": episode_nb,
                    "Episode Running Score": float(running_score),
                    "Episode Score": float(score),
                    "Episode Steps": float(steps),
                    "Epsilon": self._agent.epsilon(),
                }
            )

            if (episode_nb + 1) % self._eval_every == 0:
                eval_score = self.eval(episode_nb)
                if eval_score >= self._best_score:
                    self._agent.save_param(
                        episode_nb, path=f"param/best_{self._model_name}.pkl"
                    )
                    self._best_score = eval_score

            if (episode_nb + 1) % self._checkpoint_every == 0:
                self._agent.save_param(
                    episode_nb, path=f"param/checkpoint_{self._model_name}.pkl"
                )

            if running_score > self._env.reward_threshold:
                print(
                    "Solved! Running reward is now {} and the last episode runs to {}!".format(
                        running_score, score
                    )
                )
                self._agent.save_param(
                    episode_nb, path=f"param/best_{self._model_name}.pkl"
                )
                break

    def eval(self, episode_nb: int, mode: str='train'):
        assert mode in ['train', 'test']
        mean_score = 0
        mean_uncert = np.array([0, 0], dtype=np.float64)
        mean_steps = 0

        for episode in range(self._eval_episodes):
            ob_t = self._eval_env.reset()
            score = 0
            steps = 0
            die = False

            while not die:
                action, _ = self._agent.select_action(ob_t, greedy=True)
                ob_t1, reward, _, die = self._eval_env.step(action)
                ob_t = ob_t1
                score += reward
                steps += 1

            uncert = np.array([0] * (2 * steps))
            save_uncert(
                episode_nb,
                episode,
                score,
                uncert,
                file=f"uncertainties/{mode}/{self._model_name}.txt",
                sigma=self._eval_env.random_noise,
            )
            mean_uncert += np.mean(uncert, axis=0) / self._eval_episodes
            mean_score += score / self._eval_episodes
            mean_steps += steps / self._eval_episodes

        print(
            "Evaluation Mean Steps: %4d | Mean Reward: %4d" % (mean_steps, mean_score)
        )
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

        return mean_score
