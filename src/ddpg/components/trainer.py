import numpy as np
import wandb
from tqdm import tqdm

from components.agent import Agent
from shared.utils.utils import save_uncert
from shared.components.env import Env
from utilities.noise import BaseNoise, OUNoise

# TODO: offpolicy training
class Trainer:
    def __init__(
        self,
        env: Env,
        eval_env: Env,
        agent: Agent,
        steer_noise,
        acc_noise,
        nb_training_ep: int,
        eval_episodes: int = 3,
        eval_every: int = 10,
        skip_zoom=None,
        checkpoint_every: int = 10,
        model_name='ddpg',
    ) -> None:
        self._env = env
        self._eval_env = eval_env
        self._agent = agent
        self._model_name = model_name
        self._steer_noise = steer_noise
        self._acc_noise = acc_noise
        self._nb_training_ep = nb_training_ep
        self._eval_episodes = eval_episodes
        self._eval_every = eval_every
        self._skip_zoom = skip_zoom
        self._checkpoint_every = checkpoint_every

        self._best_score = -100
        self._eval_nb = 0
        self._global_step = 0

    def run(self):
        running_score = 0

        for episode_nb in tqdm(range(self._nb_training_ep), "Training"):
            self._steer_noise.reset()
            self._acc_noise.reset()
            ob_t = self._env.reset()
            score = 0
            rewards = []
            steps = 0
            green_rewards = []
            base_rewards = []
            speeds = []

            if self._skip_zoom is not None:
                for _ in range(self._skip_zoom):
                    ob_t = self._env.step([0, 0, 0])[0]

            for _ in range(1000):
                action = self._agent.select_action(ob_t, self._steer_noise, self._acc_noise)
                ob_t1, reward, done, die, info = self._env.step(action)
                if self._agent.store_transition(
                    ob_t, action, ob_t1, reward, (done or die)
                ):
                    self._agent.update()

                to_log = {
                    "Instant Step": self._global_step,
                    "Instant Score": float(reward),
                    "Instant Green Reward": float(info["green_reward"]),
                    "Instant Base Reward": float(info["base_reward"]),
                    "Instant Mean Speed": float(info["speed"]),
                    "Instant Noise": float(info["noise"]),
                }
                if isinstance(self._steer_noise, BaseNoise):
                    to_log["Instant Steer Noise std"] = self._steer_noise.std
                elif isinstance(self._steer_noise, OUNoise):
                    to_log["Instant Steer Noise"] = self._steer_noise.get_state()[0]
                if isinstance(self._acc_noise, BaseNoise):
                    to_log["Instant Acc Noise std"] = self._acc_noise.std
                elif isinstance(self._acc_noise, OUNoise):
                    to_log["Instant Acc Noise"] = self._acc_noise.get_state()[0]
                wandb.log(to_log)

                score += reward
                rewards.append(reward)
                ob_t = ob_t1
                steps += 1
                green_rewards.append(info["green_reward"])
                speeds.append(info["speed"])
                base_rewards.append(info["base_reward"])
                self._global_step += 1

                if done or die:
                    break

            running_score = running_score * 0.99 + score * 0.01
            wandb.log(
                {
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
        
        self._env.close()
        self._eval_env.close()

    def eval(self, episode_nb: int, mode: str = "train"):
        assert mode in ["train", "test"]
        mean_score = 0
        mean_uncert = np.array([0, 0], dtype=np.float64)
        mean_steps = 0

        for episode in tqdm(range(self._eval_episodes), f'Evaluating ep {episode_nb}'):
            ob_t = self._eval_env.reset()
            score = 0
            steps = 0
            die = False

            while not die:
                action = self._agent.select_action(ob_t)
                ob_t1, reward, _, die, _ = self._eval_env.step(action)
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

        # print(
        #     "Evaluation Mean Steps: %4d | Mean Reward: %4d" % (mean_steps, mean_score)
        # )
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
