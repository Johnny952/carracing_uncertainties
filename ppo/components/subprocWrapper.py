import multiprocessing as mp
import numpy as np

from components import Env


class SubprocessEnv(object):
    def __init__(self, nb_subp, agent, img_stack, action_repeat, eval=False, render=False, validations=3):
        self._nb_subp = nb_subp

        self._agent = agent
        self._envs = [
            Env(
                img_stack=img_stack,
                action_repeat=action_repeat,
                seed=None,
                path_render='' if render else None,
                validations=validations
                ) for _ in range(nb_subp)
        ]
        self._runs = 0
        self._eval = eval



    def _env_run(self, env, scores, uncertainties):
        score = 0
        state = env.reset()
        done = False
        die = False
        uncert = []

        while not (done or die):
            action, a_logp, (epis, aleat) = self._agent.select_action(state)
            uncert.append([epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]])
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if not self._eval:
                self._agent.store_transition((state, action, a_logp, reward, state_))
            score += reward
            state = state_

        self._runs += 1
        scores.put(score)
        uncertainties.put(uncert)
    
    def main(self):
        procs = []
        scores = mp.Queue(self._nb_subp)
        uncertainties = mp.Queue(self._nb_subp)
        for env in self._envs:
            proc = mp.Process(target=self._env_run, args=(env, scores, uncertainties))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        return scores, uncertainties