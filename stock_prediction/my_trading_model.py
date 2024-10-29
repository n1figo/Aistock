# environment.py

import numpy as np

class StockTradingEnv:
    def __init__(self, data):
        self.data = data  # 주가 데이터 (numpy 배열)
        self.n_steps = len(data)
        self.current_step = 0
        self.initial_cash = 10000  # 초기 자본
        self.cash = self.initial_cash
        self.stock_holdings = 0
        self.total_asset = self.initial_cash
        self.done = False

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.stock_holdings = 0
        self.total_asset = self.initial_cash
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        # 현재 상태 반환: [현재 주가, 보유 주식 수, 현금 잔고]
        current_price = float(self.data[self.current_step])
        obs = np.array([
            current_price,
            self.stock_holdings,
            self.cash
        ], dtype=float)
        return obs

    def step(self, action):
        """
        action: 0 (매수), 1 (매도), 2 (관망)
        """
        current_price = float(self.data[self.current_step])

        if action == 0:  # 매수
            # 최대한 매수할 수 있는 주식 수 계산
            max_shares = self.cash // current_price
            if max_shares > 0:
                self.stock_holdings += max_shares
                self.cash -= max_shares * current_price
        elif action == 1:  # 매도
            if self.stock_holdings > 0:
                self.cash += self.stock_holdings * current_price
                self.stock_holdings = 0
        # 관망(2)일 경우 아무 것도 하지 않음

        self.current_step += 1

        # 에피소드 종료 조건
        if self.current_step >= self.n_steps - 1:
            self.done = True

        # 총 자산 계산
        self.total_asset = self.cash + self.stock_holdings * current_price

        # 보상 계산: 총 자산의 변화량
        reward = self.total_asset - self.initial_cash

        # 다음 상태 관찰
        next_obs = self._get_observation()

        return next_obs, reward, self.done, {}
