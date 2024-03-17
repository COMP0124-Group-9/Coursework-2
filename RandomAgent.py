import numpy

import TrainedAgent


class RandomAgent(TrainedAgent.TrainedAgent):
    def action(self, observation, info):
        return numpy.random.choice(self._possible_actions)
