'''
Converts predictions into trading signals
'''

import pandas as pd
import numpy as np
from backtest import Backtest

class SignalGenerator:
    def __init__(self, df_combined = None, lr_pred_test = None, xgb_pred_test = None, X_test = None, y_test = None, X_test_scaled = None, y_train = None, X_train_scaled = None, X_train = None):
        self.df_combined = pd.read_csv('df_combined.csv')
        self.lr_pred_test = pd.read_csv('data/lr_pred_test.csv')
        self.xgb_pred_test = pd.read_csv('data/xgb_pred_test.csv')
        self.X_test = pd.read_csv('data/X_test.csv')
        self.y_test = pd.read_csv('data/y_test.csv')
        self.X_test_scaled = pd.read_csv('data/X_test_scaled.csv')

    def aggregate_results(self, ticker_results_dict):
        """Aggregate results across tickers"""
        total_initial = sum(r['initial_capital'] for r in ticker_results_dict.values())
        total_final = sum(r['final_capital'] for r in ticker_results_dict.values())
        total_return = (total_final - total_initial) / total_initial
        
        # Weighted average Sharpe (by capital)
        weighted_sharpe = sum(
            r['sharpe_ratio'] * r['initial_capital'] 
            for r in ticker_results_dict.values()
        ) / total_initial if total_initial > 0 else 0
        
        # Average max drawdown
        avg_drawdown = np.mean([r['max_drawdown'] for r in ticker_results_dict.values()])
        
        # Total trades
        total_trades = sum(r.get('num_trades', 0) for r in ticker_results_dict.values())
        
        return {
            'total_initial': total_initial,
            'total_final': total_final,
            'total_return': total_return,
            'weighted_sharpe': weighted_sharpe,
            'avg_drawdown': avg_drawdown,
            'total_trades': total_trades
        }

    def run_back(self):

        # Per-Ticker Backtesting
        # Split test data by ticker and backtest each separately

        # Prepare test data with predictions
        df_test_backtest = self.df_combined.loc[self.X_test.index].copy()
        df_test_backtest = df_test_backtest.sort_values('date').reset_index(drop=True)

        # Add predictions to test dataframe
        df_test_backtest['lr_pred'] = self.lr_pred_test
        df_test_backtest['xgb_pred'] = self.xgb_pred_test

        # Get unique tickers
        unique_tickers = df_test_backtest['ticker'].unique()
        print(f"Tickers to backtest: {unique_tickers}")
        print("="*80)

        # Initialize storage for results
        lr_ticker_results = {}
        xgb_ticker_results = {}
        bh_ticker_results = {}

        lr_ticker_equity = {}
        xgb_ticker_equity = {}
        bh_ticker_equity = {}

        initial_capital_per_ticker = 10000  # Capital allocated per ticker

        # Backtest each ticker separately
        for ticker in unique_tickers:
            print(f"\nBacktesting {ticker}...")
            
            # Get data for this ticker only
            df_ticker = df_test_backtest[df_test_backtest['ticker'] == ticker].copy()
            df_ticker = df_ticker.sort_values('date').reset_index(drop=True)
            
            if len(df_ticker) == 0:
                print(f"  No test data for {ticker}, skipping...")
                continue
            
            # Get signals for this ticker
            lr_signals_ticker = df_ticker['lr_pred'].values
            xgb_signals_ticker = df_ticker['xgb_pred'].values
            
            # Run backtests for this ticker
            lr_res, lr_eq = Backtest.backtest_strategy(
                df_ticker,
                pd.Series(lr_signals_ticker, index=df_ticker.index),
                initial_capital=initial_capital_per_ticker
            )
            
            xgb_res, xgb_eq = Backtest.backtest_strategy(
                df_ticker,
                pd.Series(xgb_signals_ticker, index=df_ticker.index),
                initial_capital=initial_capital_per_ticker
            )
            
            bh_res, bh_eq = Backtest.buy_and_hold(df_ticker, initial_capital=initial_capital_per_ticker)
            
            # Store results
            lr_ticker_results[ticker] = lr_res
            xgb_ticker_results[ticker] = xgb_res
            bh_ticker_results[ticker] = bh_res
            
            lr_ticker_equity[ticker] = lr_eq
            xgb_ticker_equity[ticker] = xgb_eq
            bh_ticker_equity[ticker] = bh_eq
            
            print(f"  LR: Return={lr_res['total_return']*100:.2f}%, Sharpe={lr_res['sharpe_ratio']:.2f}, Trades={lr_res['num_trades']}")
            print(f"  XGB: Return={xgb_res['total_return']*100:.2f}%, Sharpe={xgb_res['sharpe_ratio']:.2f}, Trades={xgb_res['num_trades']}")
            print(f"  B&H: Return={bh_res['total_return']*100:.2f}%, Sharpe={bh_res['sharpe_ratio']:.2f}")

        # Aggregate results across all tickers
        print("\n" + "="*80)
        print("AGGREGATE RESULTS (Sum across all tickers)")
        print("="*80)

        # lr_ticker_results

        lr_aggregate = self.aggregate_results(lr_ticker_results)
        xgb_aggregate = self.aggregate_results(xgb_ticker_results)
        bh_aggregate = self.aggregate_results(bh_ticker_results)

        print(f"\n{'Metric':<25} {'Logistic Regression':<25} {'XGBoost':<25} {'Buy & Hold':<25}")
        print("-"*100)
        print(f"{'Total Initial Capital':<25} ${lr_aggregate['total_initial']:>23,.2f} ${xgb_aggregate['total_initial']:>23,.2f} ${bh_aggregate['total_initial']:>23,.2f}")
        print(f"{'Total Final Capital':<25} ${lr_aggregate['total_final']:>23,.2f} ${xgb_aggregate['total_final']:>23,.2f} ${bh_aggregate['total_final']:>23,.2f}")
        print(f"{'Total Return':<25} {lr_aggregate['total_return']*100:>23.2f}% {xgb_aggregate['total_return']*100:>23.2f}% {bh_aggregate['total_return']*100:>23.2f}%")
        print(f"{'Weighted Avg Sharpe':<25} {lr_aggregate['weighted_sharpe']:>23.2f} {xgb_aggregate['weighted_sharpe']:>23.2f} {bh_aggregate['weighted_sharpe']:>23.2f}")
        print(f"{'Average Max Drawdown':<25} {lr_aggregate['avg_drawdown']*100:>23.2f}% {xgb_aggregate['avg_drawdown']*100:>23.2f}% {bh_aggregate['avg_drawdown']*100:>23.2f}%")
        print(f"{'Total Number of Trades':<25} {lr_aggregate['total_trades']:>23} {xgb_aggregate['total_trades']:>23} {'N/A':>23}")

        # Per-ticker performance table
        print("\n" + "="*80)
        print("PER-TICKER PERFORMANCE")
        print("="*80)

        per_ticker_df = pd.DataFrame({
            'Ticker': unique_tickers,
            'LR Return (%)': [lr_ticker_results[t]['total_return']*100 for t in unique_tickers],
            'LR Sharpe': [lr_ticker_results[t]['sharpe_ratio'] for t in unique_tickers],
            'LR Trades': [lr_ticker_results[t]['num_trades'] for t in unique_tickers],
            'XGB Return (%)': [xgb_ticker_results[t]['total_return']*100 for t in unique_tickers],
            'XGB Sharpe': [xgb_ticker_results[t]['sharpe_ratio'] for t in unique_tickers],
            'XGB Trades': [xgb_ticker_results[t]['num_trades'] for t in unique_tickers],
            'B&H Return (%)': [bh_ticker_results[t]['total_return']*100 for t in unique_tickers],
            'B&H Sharpe': [bh_ticker_results[t]['sharpe_ratio'] for t in unique_tickers]
        })

        print(per_ticker_df.to_string(index=False))
                
