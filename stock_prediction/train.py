# train.py

import numpy as np
from environment import StockTradingEnv
from agent import DQNAgent
import yfinance as yf
import torch
import os

def train_agent(ticker):
    # 모델 저장 폴더 생성
    if not os.path.exists('models'):
        os.makedirs('models')

    # 주가 데이터 수집
    data = yf.download(ticker, period='5y')['Close'].values

    # 데이터가 충분한지 확인
    if len(data) < 60:
        print("Not enough data to train the model.")
        return

    env = StockTradingEnv(data)
    state_size = 3  # 상태 크기 (현재 주가, 보유 주식 수, 현금 잔고)
    action_size = 3  # 행동 크기 (매수, 매도, 관망)
    agent = DQNAgent(state_size, action_size)
    episodes = 50  # 학습 에피소드 수
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            state = np.reshape(state, [1, state_size])
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    # 학습된 모델 저장
    model_path = os.path.join('models', f"{ticker}_dqn_model.pth")
    torch.save(agent.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
