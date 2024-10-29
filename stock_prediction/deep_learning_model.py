# deep_learning_model.py

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import time

class StockPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet.transpose()
    income_statement = stock.financials.transpose()
    cash_flow = stock.cashflow.transpose()

    # 필요한 재무 지표 계산
    financial_data = pd.DataFrame()
    financial_data['Date'] = balance_sheet.index
    financial_data['Total Assets'] = balance_sheet['Total Assets']
    financial_data['Total Liab'] = balance_sheet['Total Liab']
    financial_data['Total Revenue'] = income_statement['Total Revenue']
    financial_data['Net Income'] = income_statement['Net Income']
    financial_data['Operating Cash Flow'] = cash_flow['Total Cash From Operating Activities']
    financial_data['Debt Ratio'] = financial_data['Total Liab'] / financial_data['Total Assets']
    financial_data['ROE'] = financial_data['Net Income'] / (financial_data['Total Assets'] - financial_data['Total Liab'])
    financial_data.reset_index(drop=True, inplace=True)
    return financial_data

def get_historical_prices(ticker):
    data = yf.download(ticker, period='max')
    prices = data[['Close']]
    prices.reset_index(inplace=True)
    prices['Date'] = prices['Date'].dt.strftime('%Y-%m-%d')
    return prices

def prepare_dataset(financial_data, prices):
    # 데이터 병합
    data = pd.merge(financial_data, prices, on='Date')
    data = data.dropna()

    # 입력(features)과 출력(target) 분리
    X = data.drop(['Date', 'Close'], axis=1).values
    y = data['Close'].values

    # 데이터 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # LSTM 입력 형태로 변환 (samples, time_steps, features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # 훈련/검증/테스트 세트 분할
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

def train_deep_learning_model(ticker, training_status, status_lock):
    try:
        financial_data = get_financial_data(ticker)
        prices = get_historical_prices(ticker)
        if financial_data.empty or prices.empty:
            with status_lock:
                training_status[ticker] = {
                    'status': f'Failed to retrieve data for {ticker}.',
                    'progress': 0,
                    'estimated_time_remaining': 'N/A'
                }
            return

        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = prepare_dataset(financial_data, prices)

        input_size = X_train.shape[2]
        hidden_size = 50
        num_layers = 2
        output_size = 1
        num_epochs = 50

        model = StockPricePredictor(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        start_time = time.time()

        for epoch in range(num_epochs):
            model.train()
            outputs = model(torch.Tensor(X_train))
            optimizer.zero_grad()
            loss = criterion(outputs, torch.Tensor(y_train))
            loss.backward()
            optimizer.step()

            # 검증 손실 계산
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.Tensor(X_val))
                val_loss = criterion(val_outputs, torch.Tensor(y_val))

            # 진행률과 남은 시간 계산
            elapsed_time = time.time() - start_time
            progress = int((epoch + 1) / num_epochs * 100)
            average_time_per_epoch = elapsed_time / (epoch + 1)
            estimated_time_remaining = average_time_per_epoch * (num_epochs - (epoch + 1))

            # 학습 상태 업데이트
            with status_lock:
                training_status[ticker] = {
                    'status': 'Training in progress',
                    'progress': progress,
                    'estimated_time_remaining': f"{int(estimated_time_remaining)} 초"
                }

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        # 모델 저장 폴더 생성
        if not os.path.exists('models'):
            os.makedirs('models')

        # 모델 저장
        model_path = os.path.join('models', f"{ticker}_dl_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }, model_path)
        print(f"Deep Learning Model saved to {model_path}")

        # 학습 완료 상태로 업데이트
        with status_lock:
            training_status[ticker] = {
                'status': 'Training completed',
                'progress': 100,
                'estimated_time_remaining': '0 초'
            }

    except Exception as e:
        with status_lock:
            training_status[ticker] = {
                'status': f'Error occurred: {str(e)}',
                'progress': 0,
                'estimated_time_remaining': 'N/A'
            }

def load_and_predict(ticker):
    model_path = os.path.join('models', f"{ticker}_dl_model.pth")
    if not os.path.exists(model_path):
        return f"Deep Learning model for {ticker} not found. Please train the model first."

    # 모델 및 스케일러 로드
    checkpoint = torch.load(model_path)
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']

    model = StockPricePredictor(input_size, hidden_size, num_layers, 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']

    # 마지막 입력 데이터 준비
    financial_data = get_financial_data(ticker)
    prices = get_historical_prices(ticker)
    if financial_data.empty or prices.empty:
        return f"Failed to retrieve data for {ticker}."

    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = prepare_dataset(financial_data, prices)
    last_X = X_test[-1].reshape(1, 1, -1)

    # 미래 주가 예측
    future_periods = [3, 6, 12, 36]
    predictions = {}
    for period in future_periods:
        future_prices = predict_future_prices(model, last_X, scaler_y, period)
        predictions[f'{period}_months'] = future_prices[-1]

    return [predictions['3_months'], predictions['6_months'], predictions['12_months'], predictions['36_months']]

def predict_future_prices(model, last_X, scaler_y, periods):
    model.eval()
    predictions = []
    input_data = torch.Tensor(last_X)

    for _ in range(periods):
        with torch.no_grad():
            output = model(input_data)
        predictions.append(output.item())
        # 다음 입력 데이터 준비 (예측값을 피드백)
        input_data = input_data.clone()
        input_data[0, 0, -1] = output.item()  # 마지막 feature를 예측값으로 대체

    # 스케일링 복원
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()
