import math

import torch
import torch.optim as optim
import numpy as np

from HandoverRL.classes import *
from os import path

# Global Variables

BATCH_SIZE = 64
GAMMA = 0.95

EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 10000

update_rate = 0.005
LR = 1e-3

memory = None
device = None

policy_net = None
target_net = None

optimizer = None
loss_fn = None

steps_done = 0

save_path = None

buf_len = 0
tower_0 = deque()
tower_1 = deque()


def init_module(metric_buffer_len, n_actions,
                target_net_state_dict_load_path=None,
                target_net_state_dict_save_path=None,
                replay_memory_len=10000):
    global policy_net, target_net, \
        optimizer, loss_fn, \
        memory, device, \
        save_path, \
        tower_0, tower_1, buf_len

    tower_0 = deque(maxlen=metric_buffer_len)
    tower_1 = deque(maxlen=metric_buffer_len)
    buf_len = metric_buffer_len

    save_path = target_net_state_dict_save_path

    memory = ReplayMemory(replay_memory_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = LSTMQNetwork(metric_buffer_len, 32, n_actions).to(device)
    target_net = LSTMQNetwork(metric_buffer_len, 32, n_actions).to(device)

    if target_net_state_dict_load_path is not None:
        if path.exists(target_net_state_dict_load_path):
            policy_net.load_state_dict(torch.load(target_net_state_dict_load_path, weights_only=True))

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    loss_fn = nn.SmoothL1Loss()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    next_states = torch.cat(batch.next_state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values

    with torch.no_grad():
        next_state_values = target_net(next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def select_action(state):
    global steps_done
    x = 0
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:

        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.unsqueeze(1)

    else:
        # 10% de chance de mandar handover
        a = np.zeros(9)
        return torch.tensor([[random.choice([*a, 1])]], device=device, dtype=torch.long)


acc_reward = 0
steps = 0
last_state = None
last_action = None
last_tower = None


def train_step(tower_0_metric: float, tower_1_metric: float, torre_atual: int, hysteresis=3) -> bool:
    global acc_reward, steps, \
        last_state, last_action, \
        last_tower

    tower_0.appendleft(tower_0_metric)
    tower_1.appendleft(tower_1_metric)

    if len(tower_0) < buf_len:
        return False

    steps += 1

    if torre_atual:
        dif = np.array(tower_1) - np.array(tower_0) + hysteresis
    else:
        dif = np.array(tower_0) - np.array(tower_1) + hysteresis

    # first call
    if last_state is None:
        last_state = torch.tensor(dif, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        last_state = F.normalize(last_state)
        last_action = select_action(last_state)
        return last_action.item()


    state = torch.tensor(dif, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    last_state = F.normalize(last_state)
    action = select_action(state)

    if last_action.item(): # fez handover

        if torre_atual == last_tower:  # handover rejeitado
            reward = -200
        else:  # handover accepted
            reward = np.sum(dif) / len(dif)
            if reward > 0:
                reward *= 5
            else:
                reward -= 20


    else:  # sem ter feito handover
        reward = dif[0]
        reward = reward + 10 if reward > 0 else reward


    acc_reward += reward
    reward = torch.tensor([reward])
    memory.push(last_state, last_action, state, reward)

    optimize_model()

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * update_rate + target_net_state_dict[key] * (
                1 - update_rate)
    target_net.load_state_dict(target_net_state_dict)

    if steps % 1000 == 0:
        print(steps)
        print(f"avg reward:{acc_reward / 1000}")
        if save_path is not None:
            torch.save(target_net.state_dict(), save_path)
        acc_reward = 0

    last_state = state
    last_action = action
    last_tower = torre_atual
    return action.item()


def handover_decision(tower_0_metric: float, tower_1_metric: float, torre_atual, hysteresis=3) -> bool:

    tower_0.appendleft(tower_0_metric)
    tower_1.appendleft(tower_1_metric)

    # wait enough metrics
    if (len(tower_0) < buf_len):
        return False

    if torre_atual:
        dif = np.array(tower_1) - np.array(tower_0) + hysteresis
    else:
        dif = np.array(tower_0) - np.array(tower_1) + hysteresis

    state = torch.tensor(dif, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    state = F.normalize(state)
    with torch.no_grad():
        action = policy_net(state).max(1).indices.unsqueeze(1)

    return action
