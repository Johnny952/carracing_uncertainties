from shared.components.env import Env

class NormalizedActionsEnv(Env):

    def action(self, action):
        # Steer, throttle, break in range [-1, 1]
        action[1:] = 0.5 * action[1:] + 0.5
        # Steer [-1, 1], throttle [0, 1], break [0, 1]
        return action

    def reverse_action(self, action):
        action[1:] = 2 * (action[1:] - 0.5)
        return action

    def step(self, action):
        action = self.action(action)
        return super().step(action)