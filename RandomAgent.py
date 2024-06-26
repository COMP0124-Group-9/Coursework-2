import numpy

import TrainedAgent


class RandomAgent(TrainedAgent.TrainedAgent):
    def action(self, observation):
        return numpy.random.choice(self.possible_actions)
