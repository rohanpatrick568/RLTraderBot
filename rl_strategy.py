import numpy as np
import os
from datetime import datetime, timedelta
from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting

from gymnasium.spaces import Box
from stable_baselines3 import PPO

class PPO_Strategy(Strategy):
    def initialize(self, short_window, long_window, signal_window):
        # Set initial parameters
        self.sleeptime = '1h'
        self.symbol = self.parameters['symbol']
        self.quantity = 10
        self.data_length = 50

    def initialize(self):
        # Load the pre-trained PPO model
        model_path = self.parameters.get('ppo_model.zip')
        if model_path is None or not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' is invalid or does not exist.")
        self.model = PPO.load(model_path)
     
    def on_trading_iteration(self):
        # Fetch historical price data
        bars = self.get_historical_prices(
            self.symbol, 
            length=self.data_length,  # Get 50 previous bars
            timestep=self.timestep
        )
        
        # Define observation space for the model
        low = np.full((self.data_length,), -np.inf)  # Minimum values for each element in the observation
        high = np.full((self.data_length,), np.inf)  # Maximum values for each element in the observation
        observation_space = Box(low=low, high=high, dtype=np.float32)
        
        # Use the model to predict the next action
        action, _states = self.model.predict(bars, deterministic=True)
        
        # Execute the action
        if action == 1:
            self.buy(self.symbol, self.quantity)
        elif action == 0:
            self.sell(self.symbol, self.quantity)
        else:
            pass

# Define the backtesting period
end = datetime.now()
start = end - timedelta(days=50)

# Run the backtest
PPO_Strategy.backtest(YahooDataBacktesting, start, end, parameters={'symbol': 'AAPL', 'timestep': 'hour'})
