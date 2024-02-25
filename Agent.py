import random


class Agent:
    def __init__(self):
        self.win_count = 0
        self.position = 0

    def action(self, observation, info):
        return random.randint(0, 5)
