# RLTraderBot

An experimental day-trading bot that uses Proximal Policy Optimization (PPO) to decide when to buy and sell a single stock (AAPL by default). Models are trained with Stable-Baselines3 on hourly Yahoo Finance data and run through a Lumibot strategy for backtesting.

## Contents

| File | What it does |
| --- | --- |
| `train_ppo_model.ipynb` | Downloads hourly price data with yfinance, builds a `gym-anytrading` stocks environment and trains a PPO model |
| `SB3_Algo_Tester.ipynb` | Compares Stable-Baselines3 algorithms (A2C, PPO) on the `stocks-v0` environment and plots episode rewards |
| `sb3_ppo_strategy.ipynb` | Notebook version of the Lumibot strategy below |
| `rl_strategy.py` | Lumibot `PPO_Strategy`: loads the trained model, feeds it the last 50 bars and buys or sells 10 shares per signal; runs a 50-day Yahoo backtest when executed |
| `models/ppo_model.zip` | A trained PPO model |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`rl_strategy.py` imports `ALPACA_CONFIG` from a local `config.py`, which is git-ignored so your keys never get committed. Create it next to `rl_strategy.py`:

```python
# config.py  (do not commit)
ALPACA_CONFIG = {
    "API_KEY": "your-alpaca-key",
    "API_SECRET_KEY": "your-alpaca-secret",
    "PAPER": True,
}
```

## Usage

1. Run `train_ppo_model.ipynb` to train a model and save it to `models/`.
2. Optionally run `SB3_Algo_Tester.ipynb` to compare algorithms.
3. Backtest the strategy:

```bash
python rl_strategy.py
```

Lumibot writes run logs to `logs/` (git-ignored).

## Disclaimer

Research code only. Nothing here is financial advice.
