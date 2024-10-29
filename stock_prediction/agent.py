# agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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
        # 상태 차원 조정
        state = state.reshape(self.state_size)
        next_state = next_state.reshape(self.state_size)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 행동 선택: ε-탐욕 정책 사용
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        # 미니배치 샘플링
        minibatch = random.sample(self.memory, batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # 미니배치에서 상태, 행동, 보상, 다음 상태, 완료 여부 추출
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # 텐서로 변환
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 현재 상태에서의 Q값 계산
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 다음 상태에서의 최대 Q값 계산
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 손실 계산
        loss = criterion(q_values, target_q_values.detach())

        # 역전파 및 모델 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
