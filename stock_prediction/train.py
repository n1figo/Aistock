# train.py

import numpy as np
from environment import StockTradingEnv
from agent import DQNAgent
import yfinance as yf
import torch
import os
import time

def train_agent(ticker, training_status, status_lock):
    # 모델 저장 폴더 생성
    if not os.path.exists('models'):
        os.makedirs('models')

    # 주가 데이터 수집
    data = yf.download(ticker, period='5y')['Close'].values

    if len(data) < 60:
        with status_lock:
            training_status[ticker] = {
                'status': 'Not enough data to train the model.',
                'progress': 0,
                'estimated_time_remaining': 'N/A'
            }
        return

    env = StockTradingEnv(data)
    state_size = 3
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    episodes = 50
    batch_size = 32

    start_time = time.time()

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            state = np.reshape(state, [state_size])
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # 진행률과 남은 시간 계산
        elapsed_time = time.time() - start_time
        progress = int((e + 1) / episodes * 100)
        average_time_per_episode = elapsed_time / (e + 1)
        estimated_time_remaining = average_time_per_episode * (episodes - (e + 1))

        # 학습 상태 업데이트
        with status_lock:
            training_status[ticker] = {
                'status': 'Training in progress',
                'progress': progress,
                'estimated_time_remaining': f"{int(estimated_time_remaining)} 초"
            }

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    # 학습된 모델 저장
    model_path = os.path.join('models', f"{ticker}_dqn_model.pth")
    torch.save(agent.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 학습 완료 상태로 업데이트
    with status_lock:
        training_status[ticker] = {
            'status': 'Training completed',
            'progress': 100,
            'estimated_time_remaining': '0 초'
        }
