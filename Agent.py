import numpy as np

EXPECTED_OBSERVATION_LENGTH = 141


class Agent:
    __expected_observation_length = EXPECTED_OBSERVATION_LENGTH
    __possible_actions = np.arange(0, 6)

    def __init__(self):
        self.win_count = 0
        self.position = 0

    def action(self, observation, info):
        assert observation.shape == (self.__expected_observation_length,)
        assert info == {}
        action = np.random.choice(self.__possible_actions)  # TODO Replace this with Q-learning using self.q for Q(s, a)
        assert action in self.__possible_actions
        return action

    def q(self, observation: np.ndarray, action: int):
        assert observation.shape == (self.__expected_observation_length,)
        assert action in self.__possible_actions
        nn_input = np.append(observation, action)
        assert nn_input.shape == (self.__expected_observation_length + 1,)
        utility = np.array(0)  # TODO replace this line: pass to NN and attain a single value utility
        assert utility.shape == ()
        return utility
