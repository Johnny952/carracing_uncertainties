import gym

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action[:1] = 0.5 * action[:1] + 0.5
        return action

    def reverse_action(self, action):
        action[:1] = 2 * (action[:1] - 0.5)
        return action
