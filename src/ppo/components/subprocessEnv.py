import pickle
import cloudpickle

from multiprocessing import Process, Pipe
import numpy as np
import time

from components.env import make_env


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

    def __call__(self):
        return self.x()


class AlreadySteppingError(Exception):
    pass
class NotSteppingError(Exception):
    pass


class SubprocVecEnv():
    def __init__(self, env_fns, agent):
        self.waiting = False
        self.closed = False
        no_of_envs = len(env_fns)
        self.agent = agent
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(no_of_envs)])
        self.ps = []
        
        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process(target = worker, 
                args = (wrk, rem, CloudpickleWrapper(fn), agent))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()


    def rollout_async(self):
        if self.waiting:
            raise AlreadySteppingError
        self.waiting = True

        for remote in self.remotes:
            remote.send(('rollout'))
            # remote.send(('rollout'))

    def rollout_wait(self):
        if not self.waiting:
            raise NotSteppingError
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        score, steps, uncert = zip(*results)
        return np.stack(score), np.stack(steps), np.stack(uncert)

    def rollout(self):
        self.rollout_async()
        return self.rollout_wait()

    # def reset(self):
    #     for remote in self.remotes:
    #         remote.send(('reset', None))

    #     return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send('close')
        for p in self.ps:
            p.join()
        self.closed = True


def worker(remote, parent_remote, env_fn, agent):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd = remote.recv()
        
        if cmd == 'rollout':
            score = 0
            steps = 0
            state = env.reset()
            die = False

            uncert = []
            while not die:
                action, a_logp, (epis, aleat) = agent.select_action(state, eval=True)
                state_, reward, _, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                score += reward
                state = state_
                steps += 1
                if env.evaluation:
                    uncert.append([epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]])
                else:
                    agent.store_transition((state, action, a_logp, reward, state_))

            uncert = np.array(uncert)
            remote.send((score, steps, uncert))

        elif cmd == 'close':
            remote.close()
            break

        else:
            print(f"CMD: {cmd}")
            raise NotImplementedError()