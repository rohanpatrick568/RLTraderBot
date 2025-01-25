import numpy as np
import pandas as pd

from config import ALPACA_CONFIG
from datetime import datetime, timedelta

from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting

from gymnasium.spaces import Box

from stable_baselines3 import PPO

class PPO_Strategy(Strategy):
    def initialize(self, short_window, long_window, signal_window):
        self.sleeptime = '1h'
        self.symbol = self.parameters['symbol']
        self.quantity = 10
        self.data_length = 50
        self.timestep = self.parameters.get('timestep', 'hour')
        self.model = PPO.load(self.parameters['ppo_model.zip'])
     
    def on_trading_iteration(self):
        # Fetch data
        bars = self.get_historical_prices(
            self.symbol, 
            length= self.data_length,  # Get 50 previous bars
            timestep=self.timestep
        )
        
        # Define observation space
        low = np.full((self.data_length,), -np.inf)  # Minimum values for each element in the observation
        high = np.full((self.data_length,), np.inf)  # Maximum values for each element in the observation
        observation_space = Box(low=low, high=high, dtype=np.float32)
        
        # Use model to trade
        action, _states = self.model.predict(bars, deterministic=True)
        
        # Execute trade
        if action == 1:
            self.buy(self.symbol, self.quantity)
        elif action == 0:
            self.sell(self.symbol, self.quantity)
        else:
            pass
        
        
end = datetime.now()
start = end - timedelta(days=50)

PPO_Strategy.backtest(YahooDataBacktesting, start, end, parameters={'symbol': 'AAPL', 'timestep': 'hour'})
