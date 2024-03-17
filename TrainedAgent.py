import Agent


class TrainedAgent(Agent.Agent):
    @property
    def epsilon(self):
        return 0

    @epsilon.setter
    def epsilon(self, epsilon):
        pass

    def train(self):
        pass
