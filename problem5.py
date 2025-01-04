'''
@ Author: tetean
@ Create time: 2025/1/4 23:13
@ Info:
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

N = 10
p = np.random.uniform(0.05, 0.3, size=N)
w = np.random.uniform(0.1, 1.0, size=N)
alpha = 0.9
episodes = 2000
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
replay_buffer_size = 10000
learning_rate = 0.001


# Neural Network for Q-Learning
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class MultiSourceEnv:
    def __init__(self, N, p, w):
        self.N = N
        self.p = p
        self.w = w
        self.state = np.zeros(N, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.N, dtype=np.float32)
        return self.state

    def step(self, action):
        reward = -np.sum(self.w * self.state)  # Immediate cost
        update = np.random.rand(self.N) < self.p
        for i in range(self.N):
            if i == action:
                self.state[i] = 0
            else:
                self.state[i] = min(1.0, self.state[i] + update[i])
        return self.state, reward, False


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.epsilon = epsilon_start
        self.model = QNetwork(state_dim, action_dim).float()
        self.target_model = QNetwork(state_dim, action_dim).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def train(self):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


env = MultiSourceEnv(N, p, w)
agent = DQNAgent(N, N, learning_rate, alpha)

rewards_per_episode = []
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for _ in range(100):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.store(state, action, reward, next_state)
        agent.train()
        state = next_state
        total_reward += reward
    rewards_per_episode.append(total_reward)
    agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
    if (episode + 1) % 50 == 0:
        agent.update_target()


