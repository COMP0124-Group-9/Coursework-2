import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Buffer import Buffer

EXPECTED_OBSERVATION_LENGTH = 140

class Agent:
    __expected_observation_length = EXPECTED_OBSERVATION_LENGTH
    action_selection = [12, 4, 0, 1, 3, 11]
    possible_actions = np.arange(len(action_selection))

    def __init__(self,
                 model,
                 reversed_controls: bool,
                 reward_vector: np.ndarray = np.ones(EXPECTED_OBSERVATION_LENGTH)):
        self.reversed_controls = reversed_controls
        self.__reward_vector = reward_vector

        # Counters for metrics
        self.num_ball_in_area = 0
        self.blocks_destroyed = 0
        self.bases_destroyed = 0

        self.epsilon = 1
        self.epsilon_decay = 0.99999
        self.min_epsilon = 0.1
        self.gamma = 0.9
        self.learning_rate = 0.00000001
        self.batch_size = 2**7
        self.buffer_capacity = self.batch_size*100

        # TODO later: add target network? add epsilon decay?

        self.cuda = torch.cuda.is_available()
        self.model = model
        if self.cuda:
            self.model = self.model.cuda()
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.buffer = Buffer(self.buffer_capacity)

        assert self.__reward_vector.shape == (EXPECTED_OBSERVATION_LENGTH,)

    def reset_count_metrics(self):
        self.num_ball_in_area = 0
        self.block_destroyed = 0
        self.base_destroyed = 0

    def reward(self, observation: np.ndarray, paddle_ball_weight: float = 2e1) -> np.ndarray:
        reward = (self.__reward_vector @ ((1 - observation) / 2)).sum()
        if observation[0] == 1 and np.all(observation[[33, 66, 99]] == -1):
            reward += -self.__reward_vector[0]
        assert reward.shape == ()
        return reward

    def filter_and_reverse_action(self, action):
        action = self.action_selection[action]
        if self.reversed_controls:
            return [0, 1, 2, 4, 3, 5, 7, 6, 9, 8, 10, 12, 11, 13, 15, 14, 17, 16][action]
        return action

    def action(self, observation):
        assert observation.shape == (self.__expected_observation_length,)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.possible_actions)
        else:
            with torch.no_grad():
                state = torch.Tensor(observation)
                assert state.shape == (self.__expected_observation_length,)
                if self.cuda:
                    state = state.cuda()
                action_probs = self.model(state)
                assert action_probs.shape == (1, len(self.possible_actions))
                action = action_probs.argmax().item()
        assert action in self.possible_actions
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
        loss = self.loss(next_Q, current_Q)
        print(loss)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.push((state, action, reward, next_state, done))
