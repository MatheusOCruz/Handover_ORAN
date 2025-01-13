import random
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LSTMQNetwork(nn.Module):
    def __init__(self, sequence_length, hidden_size, n_actions):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=4, batch_first=True)

        # rede conectada dps do lstm
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * sequence_length, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, n_obs)
        lstm_out, _ = self.lstm(x)
        flat = lstm_out.contiguous().view(lstm_out.size(0), -1)

        return F.softmax(self.fc(flat), dim=1)
