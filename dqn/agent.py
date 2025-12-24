import torch
import random
import numpy as np
from dqn.model import DQN
from config import *

class DQNAgent:
    def __init__(self):
        self.model = DQN(STATE_SIZE, ACTION_SIZE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.epsilon = EPSILON
        self.memory = []

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)

        with torch.no_grad():
            q = self.model(torch.tensor(state).float())
        return torch.argmax(q).item()

    def replay(self):
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)
        for s, a, r, s2, done in batch:
            target = r
            if not done:
                target += GAMMA * torch.max(self.model(torch.tensor(s2).float()))

            q_vals = self.model(torch.tensor(s).float())
            q_vals[a] = target

            loss = torch.nn.functional.mse_loss(
                self.model(torch.tensor(s).float()), q_vals
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
