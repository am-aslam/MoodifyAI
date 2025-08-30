# flappy/dqn.py
import math, random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import ReplayBuffer, Transition
from . import config

class QNetwork(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim:int, action_dim:int, device:str=config.DEVICE):
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.online = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=config.LR)
        self.memory = ReplayBuffer(config.MEM_CAPACITY)

        self.steps_done = 0
        self.action_dim = action_dim

    def select_action(self, state:np.ndarray, eps:float) -> int:
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(torch.argmax(q, dim=1).item())

    def compute_eps(self) -> float:
        # Linear or exponential decay; use exponential-like schedule
        return config.EPS_END + (config.EPS_START - config.EPS_END) *                math.exp(-1.0 * self.steps_done / config.EPS_DECAY)

    def optimize(self):
        if len(self.memory) < max(config.BATCH_SIZE, config.TRAIN_START):
            return None

        batch = self.memory.sample(config.BATCH_SIZE)
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = torch.argmax(self.online(next_state_batch), dim=1, keepdim=True)
            next_q = self.target(next_state_batch).gather(1, next_actions)
            target_q = reward_batch + (1.0 - done_batch) * config.GAMMA * next_q

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

        if self.steps_done % config.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()
