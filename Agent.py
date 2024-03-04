import numpy as np

EXPECTED_OBSERVATION_LENGTH = 141


class Agent:
    __expected_observation_length = EXPECTED_OBSERVATION_LENGTH
    __possible_actions = np.arange(0, 6)

    def __init__(self, reward_vector: np.ndarray = np.ones(EXPECTED_OBSERVATION_LENGTH)):
        self.win_count = 0
        self.position = 0
        self.__reward_vector = reward_vector
        assert self.__reward_vector.shape == (EXPECTED_OBSERVATION_LENGTH,)

    def reward(self, observation: np.ndarray) -> np.ndarray:
        reward = (self.__reward_vector @ observation).sum()
        assert reward.shape == (1,)
        return reward

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
