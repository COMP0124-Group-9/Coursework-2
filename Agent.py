import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Network import QNetworkFC
from Buffer import Buffer

EXPECTED_OBSERVATION_LENGTH = 119

class Agent:
    __expected_observation_length = EXPECTED_OBSERVATION_LENGTH
    __possible_actions = np.arange(0, 6)

    def __init__(self, reward_vector: np.ndarray = np.ones(EXPECTED_OBSERVATION_LENGTH)):
        self.win_count = 0
        self.position = 0
        self.__reward_vector = reward_vector

        self.epsilon = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.01
        self.batch_size = 64
        self.buffer_capacity = 10000

        # TODO later: add target network? add epsilon decay?

        self.cuda = torch.cuda.is_available()
        self.model = QNetworkFC(EXPECTED_OBSERVATION_LENGTH, len(self.__possible_actions))
        if self.cuda:
            self.model = self.model.cuda()
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.buffer = Buffer(self.buffer_capacity)

        assert self.__reward_vector.shape == (EXPECTED_OBSERVATION_LENGTH,)

    def reward(self, observation: np.ndarray) -> np.ndarray:
        reward = (self.__reward_vector @ observation).sum()
        assert reward.shape == ()
        return reward

    def action(self, observation, info):
        assert observation.shape == (self.__expected_observation_length,)
        assert info == {}
        if np.random.rand() < self.epsilon:
            action =  np.random.choice(self.__possible_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0)
                if self.cuda:
                    state = state.cuda()
                action = self.model(state).argmax().item()
        assert action in self.__possible_actions
        return action

    def train(self):

        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(np.array(batch[0]))
        action_batch = torch.LongTensor(np.array(batch[1])).view(-1, 1)
        reward_batch = torch.FloatTensor(np.array(batch[2])).view(-1, 1)
        next_state_batch = torch.FloatTensor(np.array(batch[3]))
        done_batch = torch.FloatTensor(np.array(batch[4])).view(-1, 1)
        if self.cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
            done_batch = done_batch.cuda()
        current_Q = self.model(state_batch).gather(1, action_batch)
        next_Q = reward_batch + (1 - done_batch) * self.gamma * self.model(next_state_batch).max(1)[0].view(-1, 1)
        loss = self.loss(current_Q, next_Q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.push((state, action, reward, next_state, done))

    def q(self, observation: np.ndarray, action: int):
        assert observation.shape == (self.__expected_observation_length,)
        assert action in self.__possible_actions
        nn_input = np.append(observation, action)
        assert nn_input.shape == (self.__expected_observation_length + 1,)
        utility = np.array(0)  # TODO replace this line: pass to NN and attain a single value utility
        assert utility.shape == ()
        return utility
