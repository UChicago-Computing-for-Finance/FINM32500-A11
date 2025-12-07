'''
Simulates trading strategy and computes performance
'''

import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, df_combined = None):
        self.df_combined = df_combined

    # Signal Generation Function
    def generate_signals(predictions, threshold=0.5):
        signals = predictions.copy()
        return signals

    # Backtesting Function
    def backtest_strategy(df, signals, initial_capital=10000, position_size=1.0):
        
        # Initialize tracking variables
        capital = initial_capital
        position = 0  # Number of shares held
        equity_curve = []
        trades = []
        
        # Align signals with dataframe
        df_backtest = df.copy()
        df_backtest['signal'] = signals
        df_backtest = df_backtest.sort_values('date').reset_index(drop=True)
        
        for i in range(len(df_backtest)):
            current_price = df_backtest.loc[i, 'close']
            current_date = df_backtest.loc[i, 'date']
            signal = df_backtest.loc[i, 'signal']
            
            # Execute trades based on signal
            if signal == 1 and position == 0:  # Buy signal, no position
                # Buy
                shares_to_buy = int((capital * position_size) / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    position = shares_to_buy
                    capital -= cost
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'capital': capital
                    })
            
            elif signal == 0 and position > 0:  # Sell signal, have position
                # Sell
                proceeds = position * current_price
                capital += proceeds
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'capital': capital
                })
                position = 0
            
            # Calculate current equity (cash + position value)
            current_equity = capital + (position * current_price)
            equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'capital': capital,
                'position': position,
                'price': current_price
            })
        
        # Close any remaining position at the end
        if position > 0:
            final_price = df_backtest.loc[len(df_backtest)-1, 'close']
            capital += position * final_price
            position = 0
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Calculate performance metrics
        total_return = (capital - initial_capital) / initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        
        # Calculate Sharpe ratio (annualized, assuming 252 trading days)
        if equity_df['returns'].std() > 0:
            sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'final_equity': final_equity,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len([t for t in trades if t['action'] == 'BUY']),
            'trades': trades
        }
        
        return results, equity_df

    # Buy-and-Hold Baseline
    def buy_and_hold(df, initial_capital=10000):
        """
        Calculate buy-and-hold strategy performance.
        """
        df_bh = df.copy().sort_values('date').reset_index(drop=True)
        
        # Buy at first price
        first_price = df_bh.loc[0, 'close']
        shares = int(initial_capital / first_price)
        initial_investment = shares * first_price
        
        # Calculate equity curve
        equity_curve = []
        for i in range(len(df_bh)):
            current_price = df_bh.loc[i, 'close']
            current_equity = shares * current_price
            equity_curve.append({
                'date': df_bh.loc[i, 'date'],
                'equity': current_equity,
                'price': current_price
            })
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        if equity_df['returns'].std() > 0:
            sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': 1  # Buy-and-hold has 1 trade (initial buy)
        }
        
        return results, equity_df


