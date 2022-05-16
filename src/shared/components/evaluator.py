import numpy as np
import os

from ppo.components.agent import Agent
from shared.components.env import Env
from shared.utils.utils import save_uncert, init_uncert_file

class Evaluator:
    def __init__(self, img_stack, action_repeat, model_name, device='cpu', validations=1, seed=123, base_path='uncertainties/customeval') -> None:
        self.validations = validations
        self._agent = Agent(
            1,
            img_stack,
            0,
            model='base',
            device=device
        )
        self._agent.load('../shared/components/controller_ppo.pkl', eval_mode=True)

        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.seed = seed
        self.validations = validations
        self.model_name = model_name
        self.base_path = base_path
        self.noise = None
        self.evaluation_nb = 0

        if not os.path.exists(base_path):
            os.makedirs(base_path)
        init_uncert_file(file=f"{self.base_path}/{self.model_name}.txt")
    
    def load_env(self):
        # TODO: Validar que todos las validaciones son iguales siempre
        self._eval_env = Env(
            img_stack=self.img_stack,
            action_repeat=self.action_repeat,
            seed=self.seed,
            validations=self.validations,
            noise=self.noise,
            path_render=f"../shared/components/render/{self.evaluation_nb}",
        )
    
    def set_noise_value(self, noise):
        self.noise = noise
    
    def eval(self, episode_nb, agent):
        self.load_env()
        for i_val in range(self.validations):
            score = 0
            steps = 0
            state = self._eval_env.reset()
            die = False

            uncert = []
            i_step = 0
            while not die:
                action = self._agent.select_action(state, eval=True)[0]
                epis, aleat = agent.select_action(state, eval=True)[-1]
                uncert.append(
                    [epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]]
                )
                action = self.ppo_step(action, i_step)

                state_, reward, _, die = self._eval_env.step(action)[:4]
                score += reward
                state = state_
                steps += 1
                i_step += 1

            uncert = np.array(uncert)
            save_uncert(
                episode_nb,
                i_val,
                score,
                uncert,
                file=f"{self.base_path}/{self.model_name}.txt",
                sigma=self._eval_env.random_noise,
            )

            self.evaluation_nb += 1
        self._eval_env.close()
        #self._eval_env.reset()
    
    def ppo_step(self, action, step):
        # TODO: Change action when step is x
        if step in [70, 71]:
            return np.array([-1.0, 0.4, 0])
        return action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])