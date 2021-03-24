import os
import psutil

import gym
import numpy as np

print(gym.__version__)

env = gym.make('CarRacing-v0')
running_score = 0

for i in range(10000):

    state = env.reset()

    process = psutil.Process(os.getpid())
    print("Memory usage at {} epoch: {} GB".format(i, process.memory_info().rss/1e9))
    print("Memory usage {}%".format(psutil.virtual_memory().percent))
    print("CPU usage {}".format(psutil.cpu_percent()))

    if process.memory_info().rss >= 1e10:   # if memory usage is over 10GB
        break

    score = 0
    for t in range(1000):
        state_, reward, done, die = env.step(np.array([1., 0., 0.]))
        score += reward
        state = state_
        if done or die:
            break
    #env.close()    # Con esto entrena el doble de iteraciones, pero el memory leak persiste
    running_score = running_score * 0.99 + score * 0.01

env.close()