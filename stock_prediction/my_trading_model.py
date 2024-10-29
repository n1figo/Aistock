# my_trading_model.py

import torch
import numpy as np
import yfinance as yf
from agent import DQNAgent
from environment import StockTradingEnv
import os

def predict(ticker, period):
    state_size = 3
    action_size = 3

    # 모델 경로 설정
    model_path = os.path.join('models', f"{ticker}_dqn_model.pth")

    # 모델 로드
    agent = DQNAgent(state_size, action_size)
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
    else:
        return f"Model for {ticker} not found. Please train the model first."

    # 예측 기간 동안의 데이터 수집
    data = yf.download(ticker, period=f'{int(period)*2}mo')['Close'].values

    # 데이터 충분성 확인
    if len(data) < int(period):
        return f"Not enough data to make predictions for {period} months."

    env = StockTradingEnv(data)
    state = env.reset()
    total_assets = []

    while not env.done:
        state = np.reshape(state, [state_size])
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        total_assets.append(env.total_asset)
        state = next_state

    # 예측 결과 반환
    return total_assets[:int(period)]

def backtest(ticker, period):
    # 과거 데이터 수집
    data = yf.download(ticker, period=f'{int(period)*2}mo')['Close'].values
    state_size = 3
    action_size = 3

    # 모델 경로 설정
    model_path = os.path.join('models', f"{ticker}_dqn_model.pth")

    # 모델 로드
    agent = DQNAgent(state_size, action_size)
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
    else:
        return {
            'predicted_assets': [],
            'total_return': 0,
            'message': f"Model for {ticker} not found. Please train the model first."
        }

    env = StockTradingEnv(data)
    state = env.reset()
    predicted_assets = []
    initial_asset = env.total_asset

    while not env.done:
        state = np.reshape(state, [state_size])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        predicted_assets.append(env.total_asset)
        state = next_state

        # 디버깅 출력 추가
        print(f"Step: {env.current_step}, Action: {action}, Total Asset: {env.total_asset}")

    # 수익률 계산
    final_asset = env.total_asset
    total_return = (final_asset - initial_asset) / initial_asset * 100

    # 자산 가치 출력
    print(f"Initial Asset: {initial_asset}, Final Asset: {final_asset}, Total Return: {total_return}%")

    return {
        'predicted_assets': predicted_assets[:int(period)],
        'total_return': total_return,
        'message': "Backtest completed."
    }
