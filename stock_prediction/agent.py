# agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 상태 공간의 크기
        self.action_size = action_size  # 행동 공간의 크기
        self.memory = deque(maxlen=2000)  # 리플레이 메모리
        self.gamma = 0.95    # 할인 인자
        self.epsilon = 1.0   # 탐험률 초기값
        self.epsilon_min = 0.01  # 탐험률 최소값
        self.epsilon_decay = 0.995  # 탐험률 감소율
        self.learning_rate = 0.001  # 학습률
        self.model = self._build_model()  # 신경망 모델

    def _build_model(self):
        # 심층 신경망 모델 정의
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        # 경험을 메모리에 저장
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 행동 선택: ε-탐욕 정책 사용
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        # 미니배치 샘플링
        minibatch = random.sample(self.memory, batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f = target_f.clone().detach()
            target_f[action] = target

            optimizer.zero_grad()
            outputs = self.model(state)
            loss = criterion(outputs, target_f)
            loss.backward()
            optimizer.step()

        # 탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
