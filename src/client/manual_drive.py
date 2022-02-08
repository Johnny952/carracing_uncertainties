import argparse
from pyglet.window import key
import numpy as np

import sys
sys.path.append('..')
from shared.components.env import Env

TURN_VALUE = 0.3
ACCELERATE_VALUES = [0, 0.3]
BRAKE_VALUES = [0.01, 0.1]

def main(env, nb_episodes):
    def reset_a():
        return np.array([0.0, ACCELERATE_VALUES[0], BRAKE_VALUES[0]]).astype('float32')
    
    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -TURN_VALUE
        if k == key.RIGHT: a[0] = +TURN_VALUE
        if k == key.UP:    a[1] = +ACCELERATE_VALUES[1]
        if k == key.DOWN:  a[2] = +BRAKE_VALUES[1]

    def key_release(k, mod):
        if k == key.LEFT: a[0] = 0.0
        if k == key.RIGHT: a[0] = 0.0
        if k == key.UP:    a[1] = ACCELERATE_VALUES[0]
        if k == key.DOWN:  a[2] = BRAKE_VALUES[0]

    env.reset()
    env.env.viewer.window.on_key_press = key_press
    env.env.viewer.window.on_key_release = key_release

    a = reset_a()
    
    episode_rewards = []
    steps = 0
    for _ in range(nb_episodes):
        episode_reward = 0
        done = False
        state = env.reset()
        while not done:
            next_state, r, _, done = env.step(a)[:4]
            episode_reward += r

            state = next_state
            steps += 1

            if steps % 1000 == 0 or done:
                print("\nstep {}".format(steps))

            env.render()
        
        episode_rewards.append(episode_reward)

    print(f"Episode score: {np.sum(episode_rewards)}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-NE",
        "--nb-episodes",
        type=int,
        default=1,
        help="Number of episodes to test",
    )
    args = parser.parse_args()

    env = Env(4, 1, evaluation=True,)

    main(env, args.nb_episodes)