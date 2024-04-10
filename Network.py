import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


####### State Input ###########

class QNetworkFC(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        size = int(4e3)
        self.network = nn.Sequential(nn.Linear(self.input_shape, size),
                                     nn.ReLU())
        for _ in range(4):
            self.network.append(nn.Linear(size, size))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(size, num_actions))

    def forward(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(dim=0)
        assert x.ndim == 2
        assert x.shape[1] == self.input_shape
        x_count = x.shape[0]
        # x = x.unsqueeze(dim=1)
        # assert x.shape == (x_count, 1, self.input_shape)
        # x = x.repeat((1, self.num_actions, 1))
        # assert x.shape == (x_count, self.num_actions, self.input_shape)
        # action_list = torch.arange(self.num_actions).reshape(1, -1, 1).repeat(x_count, 1, 1).to(x.device)
        # assert action_list.shape == (x_count, self.num_actions, 1)
        # x = torch.concatenate((x, action_list), dim=-1)
        # assert x.shape == (x_count, self.num_actions, self.input_shape + 1)
        # result = self.network(x).squeeze(dim=-1)
        result=self.network(x)
        assert result.shape == (x_count, self.num_actions)
        return result


class QNetworkCNN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(QNetworkCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x


####### Action-State Input #########

class QNetworkFCAS(nn.Module):
    def __init__(self, input_shape):
        super(QNetworkFCAS, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)  # TODO test with different architecture
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat((state.view(state.size(0), -1), action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)


class QNetworkCNNAS(nn.Module):
    def __init__(self, input_shape):
        super(QNetworkCNNAS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512 + 1, 1)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        return x.squeeze(1)


if __name__ == "__main__":

    input_shape = 141
    num_actions = 6
    learning_rate = 1e-2
    network_type = "CNN"
    input_type = "state"

    if network_type == "FCNN":
        if input_type == "state":
            network = QNetworkFC(input_shape, num_actions)
        elif input_type == "action-state":
            network = QNetworkFCAS(input_shape, num_actions)
    elif network_type == "CNN":
        if input_type == "state":
            network = QNetworkCNN(input_shape, num_actions)
        elif input_type == "action-state":
            network = QNetworkCNNAS(input_shape, num_actions)
