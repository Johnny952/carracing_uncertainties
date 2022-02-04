from shared.components.env import Env
import copy

class NormalizedActionsEnv(Env):

    def action(self, action):
        clone_action = copy.deepcopy(action)
        # Steer, throttle, break in range [-1, 1]
        clone_action[1:] = 0.5 * clone_action[1:] + 0.5
        # Steer [-1, 1], throttle [0, 1], break [0, 1]
        return clone_action

    def reverse_action(self, action):
        clone_action = copy.deepcopy(action)
        clone_action[1:] = 2 * (clone_action[1:] - 0.5)
        return clone_action

    def step(self, action):
        return super().step(self.action(action))