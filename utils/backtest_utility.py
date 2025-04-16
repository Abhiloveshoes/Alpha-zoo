import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def backtest_strategy(returns, positions):
    strategy_returns = returns * positions.shift(1)  # use previous day's signal
    strategy_returns = strategy_returns.fillna(0)
    cumulative = (1 + strategy_returns).cumprod()
    return cumulative


def market_neutral_weights(signal: pd.Series) -> pd.Series:
    """Convert signal into market-neutral weights"""
    if signal.std() == 0:
        return pd.Series(0, index=signal.index)  # Handle flat signals
    weights = signal - signal.mean()
    return weights / weights.abs().sum()

def backtest_portfolio(prices: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Compute portfolio returns"""
    daily_returns = prices.pct_change().fillna(0)
    portfolio_returns = (weights.shift(1) * daily_returns).sum(axis=1)
    return (1 + portfolio_returns).cumprod()

def plot_equity_curve(equity_curve: pd.Series, title="Equity Curve"):
    equity_curve.plot(figsize=(10, 4), title=title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

