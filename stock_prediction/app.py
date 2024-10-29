# app.py

from flask import Flask, request, jsonify, render_template
from threading import Thread, Lock
from my_trading_model import predict, backtest
from train import train_agent
from deep_learning_model import train_and_predict
import time

app = Flask(__name__)

# 학습 상태를 저장할 딕셔너리와 락
training_status = {}
status_lock = Lock()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    data = request.get_json()
    ticker = data.get('ticker')

    # 학습 작업을 새로운 스레드에서 시작
    training_thread = Thread(target=train_model_thread, args=(ticker,))
    training_thread.start()

    # 학습 상태 초기화
    with status_lock:
        training_status[ticker] = {
            'status': 'Training started',
            'progress': 0,
            'estimated_time_remaining': 'Calculating...'
        }

    return jsonify({'message': f'Model training started for {ticker}'}), 200

def train_model_thread(ticker):
    # 학습 함수 호출
    train_agent(ticker, training_status, status_lock)

@app.route('/training_status', methods=['GET'])
def get_training_status():
    ticker = request.args.get('ticker')
    with status_lock:
        status = training_status.get(ticker, None)
    if status:
        return jsonify(status), 200
    else:
        return jsonify({'status': 'No training in progress for this ticker.'}), 200

@app.route('/predict', methods=['POST'])
def predict_stock():
    data = request.get_json()
    ticker = data.get('ticker')
    period = data.get('period')

    predictions = predict(ticker, period)
    if isinstance(predictions, str):
        return jsonify({'error': predictions}), 400

    backtest_results = backtest(ticker, period)
    if 'message' in backtest_results and 'Model not found' in backtest_results['message']:
        return jsonify({'error': backtest_results['message']}), 400

    return jsonify({
        'predictions': predictions,
        'backtest': backtest_results
    }), 200

@app.route('/deep_learning_predict', methods=['POST'])
def deep_learning_predict():
    data = request.get_json()
    ticker = data.get('ticker')

    # 모델 학습 및 예측
    predictions = train_and_predict(ticker)
    if isinstance(predictions, str):
        return jsonify({'error': predictions}), 400

    return jsonify({
        '3_months': predictions[0],
        '6_months': predictions[1],
        '12_months': predictions[2],
        '36_months': predictions[3]
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
