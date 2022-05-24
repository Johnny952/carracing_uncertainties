import numpy as np
import wandb
from tqdm import tqdm

from components.uncert_agents.abstact import AbstactAgent
from shared.utils.utils import save_uncert
from shared.components.env import Env
from shared.components.evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        env: Env,
        eval_env: Env,
        agent: AbstactAgent,
        nb_training_ep: int,
        eval_episodes: int = 3,
        eval_every: int = 10,
        skip_zoom=None,
        model_name: str = "base",
        checkpoint_every: int = 10,
        evaluator: Evaluator = None,
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
        self._evaluator = evaluator

        self._best_score = -100
        self._eval_nb = 0
        self._global_step = 0

    def run(self):
        running_score = 0

        for episode_nb in tqdm(range(self._nb_training_ep), "Training"):
            ob_t = self._env.reset()
            score = 0
            steps = 0
            rewards = []
            green_rewards = []
            base_rewards = []
            speeds = []

            if self._skip_zoom is not None:
                for _ in range(self._skip_zoom):
                    ob_t = self._env.step([0, 0, 0])[0]

            for _ in range(1000):
                action, action_idx = self._agent.select_action(ob_t)[:2]
                ob_t1, reward, done, die, info = self._env.step(action)
                if self._agent.store_transition(
                    ob_t, action_idx, ob_t1, reward, (done or die)
                ):
                    self._agent.update()
                # wandb.log(
                #     {
                #         "Instant Step": self._global_step,
                #         "Instant Score": float(reward),
                #         "Instant Green Reward": float(info["green_reward"]),
                #         "Instant Base Reward": float(info["base_reward"]),
                #         "Instant Mean Speed": float(info["speed"]),
                #         "Instant Noise": float(info["noise"]),
                #     }
                # )

                score += reward
                ob_t = ob_t1
                steps += 1
                rewards.append(reward)
                green_rewards.append(info["green_reward"])
                speeds.append(info["speed"])
                base_rewards.append(info["base_reward"])
                self._global_step += 1

                if done or die:
                    break

            running_score = running_score * 0.99 + score * 0.01
            wandb.log(
                {
                    "Epsilon": self._agent.epsilon(),
                    "Train Episode": episode_nb,
                    "Episode Running Score": float(running_score),
                    "Episode Score": float(score),
                    "Episode Steps": float(steps),
                    "Episode Min Reward": float(np.min(rewards)),
                    "Episode Max Reward": float(np.max(rewards)),
                    "Episode Mean Reward": float(np.mean(rewards)),
                    "Episode Green Reward": float(np.sum(green_rewards)),
                    "Episode Base Reward": float(np.sum(base_rewards)),
                    "Episode Mean Speed": float(np.mean(speeds)),
                    "Episode Noise": float(info["noise"]),
                }
            )

            self._agent.epsilon_step()

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

    def eval(self, episode_nb: int, mode: str = "eval"):
        if self._evaluator:
            self._evaluator.eval(episode_nb, self._agent)
        return self.base_eval(episode_nb, mode=mode)

    def base_eval(self, episode_nb: int, mode: str = "train"):
        assert mode in ['train', 'eval', 'test0', 'test']
        mean_score = 0
        mean_uncert = np.array([0, 0], dtype=np.float64)
        mean_steps = 0

        for episode in tqdm(range(self._eval_episodes), f'{mode.title()} ep {episode_nb}'):
            ob_t = self._eval_env.reset()
            score = 0
            steps = 0
            die = False

            uncert = []
            while not die:
                action, _, (epis, aleat) = self._agent.select_action(ob_t, eval=True)
                ob_t1, reward, _, die = self._eval_env.step(action)[:4]
                ob_t = ob_t1
                score += reward
                steps += 1
                uncert.append(
                    [epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]]
                )

            uncert = np.array(uncert)
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

        wandb_mode = mode.title()
        wandb.log(
            {
                f"{wandb_mode} Episode": self._eval_nb,
                f"{wandb_mode} Mean Score": float(mean_score),
                f"{wandb_mode} Mean Epist Uncert": float(mean_uncert[0]),
                f"{wandb_mode} Mean Aleat Uncert": float(mean_uncert[1]),
                F"{wandb_mode} Mean Steps": float(mean_steps),
            }
        )
        self._eval_nb += 1

        return mean_score