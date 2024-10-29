# app.py

from flask import Flask, request, jsonify, render_template
from my_trading_model import predict, backtest
from train import train_agent

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    ticker = data.get('ticker')
    train_agent(ticker)
    return jsonify({'message': f'Model trained for {ticker}'}), 200

@app.route('/predict', methods=['POST'])
def predict_stock():
    data = request.get_json()
    ticker = data.get('ticker')
    period = data.get('period')

    predictions = predict(ticker, period)
    if isinstance(predictions, str):
        # 모델이 없는 경우 에러 메시지 반환
        return jsonify({'error': predictions}), 400

    backtest_results = backtest(ticker, period)
    if 'message' in backtest_results and 'Model not found' in backtest_results['message']:
        return jsonify({'error': backtest_results['message']}), 400

    return jsonify({
        'predictions': predictions,
        'backtest': backtest_results
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
