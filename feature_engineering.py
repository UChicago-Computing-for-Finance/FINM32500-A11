'''
Generates features and labels from raw data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FeatureEngineering:
    def __init__(self, df_tickers = None, df_market_data = None):
        self.df_tickers = df_tickers
        self.df_market_data = df_market_data

    def create_features(self):

        df_tickers = pd.read_csv('tickers-1.csv')
        df_market_data = pd.read_csv('market_data_ml.csv')

        # total df
        dfs = []
        for ticker in df_tickers['symbol']:
            dfs.append(df_market_data[df_market_data['ticker'] == ticker])
            globals()['df_%s' % ticker.lower()] = df_market_data[df_market_data['ticker'] == ticker]

        # create features
        for i, df in enumerate(dfs):
            df = df.copy()
            
            # Ensure date column is datetime and sort by date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate daily returns (using pandas pct_change)
            df['daily_return'] = df['close'].pct_change()
            
            # Calculate log returns (using numpy)
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Create lag features for returns (1-day, 3-day, 5-day) - using pandas shift
            df['return_lag_1'] = df['daily_return'].shift(1)
            df['return_lag_3'] = df['daily_return'].shift(3)
            df['return_lag_5'] = df['daily_return'].shift(5)
            
            # Add Simple Moving Averages (SMA) - using pandas rolling
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Add RSI using pandas/numpy
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Add MACD using pandas ewm (exponential weighted moving average)
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            df['direction'] = np.where(df['daily_return'] > 0, 1, 0)
            
            # Update the dataframe in the list
            dfs[i] = df
            
            # Also update the global variable if it exists
            ticker = df['ticker'].iloc[0]
            if f'df_{ticker.lower()}' in globals():
                globals()[f'df_{ticker.lower()}'] = df

        df_combined = pd.concat(dfs, ignore_index=True)

        # save the combined dataframe
        df_combined.to_csv('df_combined.csv', index=False)
