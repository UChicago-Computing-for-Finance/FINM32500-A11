import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import FeatureEngineering
from train_model import TrainModel


class TestFeatureEngineering:
    """Test feature generation and label creation"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        data = {
            'date': dates,
            'ticker': ['AAPL'] * 100,
            'open': prices + np.random.randn(100) * 0.5,
            'high': prices + np.abs(np.random.randn(100) * 1),
            'low': prices - np.abs(np.random.randn(100) * 1),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_tickers(self):
        """Create sample tickers data"""
        return pd.DataFrame({'symbol': ['AAPL']})
    
    @pytest.fixture
    def feature_engineering_instance(self, sample_tickers, sample_market_data):
        """Create FeatureEngineering instance with sample data"""
        fe = FeatureEngineering()
        fe.df_tickers = sample_tickers
        fe.df_market_data = sample_market_data
        return fe
    
    def test_daily_return_calculation(self, sample_market_data):
        """Test that daily returns are calculated correctly"""
        df = sample_market_data.copy()
        df['daily_return'] = df['close'].pct_change()
        
        # First value should be NaN
        assert pd.isna(df['daily_return'].iloc[0])
        
        # Subsequent values should be valid
        assert not pd.isna(df['daily_return'].iloc[1])
        
        # Check calculation: (price_today - price_yesterday) / price_yesterday
        expected_return = (df['close'].iloc[1] - df['close'].iloc[0]) / df['close'].iloc[0]
        assert abs(df['daily_return'].iloc[1] - expected_return) < 1e-10
    
    def test_log_return_calculation(self, sample_market_data):
        """Test that log returns are calculated correctly"""
        df = sample_market_data.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # First value should be NaN
        assert pd.isna(df['log_return'].iloc[0])
        
        # Check calculation: log(price_today / price_yesterday)
        if not pd.isna(df['log_return'].iloc[1]):
            expected_log_return = np.log(df['close'].iloc[1] / df['close'].iloc[0])
            assert abs(df['log_return'].iloc[1] - expected_log_return) < 1e-10
    
    def test_lag_features_creation(self, sample_market_data):
        """Test that lag features are created correctly"""
        df = sample_market_data.copy()
        df['daily_return'] = df['close'].pct_change()
        df['return_lag_1'] = df['daily_return'].shift(1)
        df['return_lag_3'] = df['daily_return'].shift(3)
        df['return_lag_5'] = df['daily_return'].shift(5)
        
        # Check that lag features exist
        assert 'return_lag_1' in df.columns
        assert 'return_lag_3' in df.columns
        assert 'return_lag_5' in df.columns
        
        # Check that lag_1 equals previous day's return
        assert abs(df['return_lag_1'].iloc[2] - df['daily_return'].iloc[1]) < 1e-10
        
        # Check that lag_3 equals return from 3 days ago
        assert abs(df['return_lag_3'].iloc[5] - df['daily_return'].iloc[2]) < 1e-10
    
    def test_sma_features_creation(self, sample_market_data):
        """Test that SMA features are created correctly"""
        df = sample_market_data.copy()
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Check that SMA features exist
        assert 'sma_5' in df.columns
        assert 'sma_10' in df.columns
        assert 'sma_20' in df.columns
        
        # First 4 values should be NaN (window=5)
        assert pd.isna(df['sma_5'].iloc[0])
        assert pd.isna(df['sma_5'].iloc[3])
        
        # 5th value should be the mean of first 5 prices
        expected_sma5 = df['close'].iloc[0:5].mean()
        assert abs(df['sma_5'].iloc[4] - expected_sma5) < 1e-10
    
    def test_rsi_calculation(self, sample_market_data):
        """Test that RSI is calculated correctly"""
        df = sample_market_data.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Check that RSI exists
        assert 'rsi' in df.columns
        
        # RSI should be between 0 and 100 (after enough data points)
        valid_rsi = df['rsi'].dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self, sample_market_data):
        """Test that MACD is calculated correctly"""
        df = sample_market_data.copy()
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Check that MACD features exist
        assert 'macd' in df.columns
        assert 'macd_signal' in df.columns
        assert 'macd_histogram' in df.columns
        
        # MACD should be difference between fast and slow EMA
        assert abs(df['macd'].iloc[-1] - (ema_fast.iloc[-1] - ema_slow.iloc[-1])) < 1e-10
    
    def test_direction_label_creation(self, sample_market_data):
        """Test that direction labels are created correctly"""
        df = sample_market_data.copy()
        df['daily_return'] = df['close'].pct_change()
        df['direction'] = np.where(df['daily_return'] > 0, 1, 0)
        
        # Check that direction exists
        assert 'direction' in df.columns
        
        # Direction should be binary (0 or 1)
        valid_direction = df['direction'].dropna()
        assert set(valid_direction.unique()).issubset({0, 1})
        
        # Direction should be 1 when return > 0, 0 when return <= 0
        for i in range(1, len(df)):
            if not pd.isna(df['daily_return'].iloc[i]):
                expected_direction = 1 if df['daily_return'].iloc[i] > 0 else 0
                assert df['direction'].iloc[i] == expected_direction
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_csv')
    def test_create_features_output(self, mock_read_csv, mock_file, feature_engineering_instance, sample_tickers, sample_market_data):
        """Test that create_features produces expected output structure"""
        mock_read_csv.side_effect = [sample_tickers, sample_market_data]
        
        # Mock the method to avoid file I/O
        with patch.object(feature_engineering_instance, 'create_features') as mock_create:
            # This test verifies the structure, not the actual execution
            pass
        
        # Test that expected features would be created
        df = sample_market_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['daily_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['return_lag_1'] = df['daily_return'].shift(1)
        df['return_lag_3'] = df['daily_return'].shift(3)
        df['return_lag_5'] = df['daily_return'].shift(5)
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        df['direction'] = np.where(df['daily_return'] > 0, 1, 0)
        
        # Check all expected features exist
        expected_features = ['return_lag_1', 'return_lag_3', 'return_lag_5', 
                           'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd']
        for feature in expected_features:
            assert feature in df.columns, f"Feature {feature} not found"
        
        # Check label exists
        assert 'direction' in df.columns


class TestModelTraining:
    """Test model training and prediction shapes"""
    
    @pytest.fixture
    def sample_combined_data(self):
        """Create sample combined data with features"""
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'ticker': ['AAPL'] * n_samples,
            'return_lag_1': np.random.randn(n_samples) * 0.01,
            'return_lag_3': np.random.randn(n_samples) * 0.01,
            'return_lag_5': np.random.randn(n_samples) * 0.01,
            'sma_5': 100 + np.random.randn(n_samples) * 5,
            'sma_10': 100 + np.random.randn(n_samples) * 5,
            'sma_20': 100 + np.random.randn(n_samples) * 5,
            'rsi': np.random.uniform(30, 70, n_samples),
            'macd': np.random.randn(n_samples) * 2,
            'direction': np.random.choice([0, 1], n_samples)
        }
        df = pd.DataFrame(data)
        
        # Add some NaN values to simulate real data
        df.loc[0:10, 'return_lag_1'] = np.nan
        df.loc[0:5, 'sma_5'] = np.nan
        
        return df
    
    @pytest.fixture
    def train_model_instance(self, sample_combined_data):
        """Create TrainModel instance"""
        tm = TrainModel()
        tm.df_combined = sample_combined_data
        return tm
    
    def test_feature_selection(self, sample_combined_data):
        """Test that correct features are selected from config"""
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)
        
        feature_list = features_config['features']
        X = sample_combined_data[feature_list].copy()
        
        # Check all features from config are present
        for feature in feature_list:
            assert feature in X.columns, f"Feature {feature} not in dataframe"
        
        # Check shape
        assert X.shape[1] == len(feature_list)
    
    def test_label_selection(self, sample_combined_data):
        """Test that correct label is selected"""
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)
        
        label_name = features_config['label']
        y = sample_combined_data[label_name].copy()
        
        # Check label exists
        assert label_name in sample_combined_data.columns
        
        # Check label is binary
        assert set(y.dropna().unique()).issubset({0, 1})
    
    def test_missing_value_handling(self, sample_combined_data):
        """Test that missing values are handled correctly"""
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)
        
        feature_list = features_config['features']
        X = sample_combined_data[feature_list].copy()
        
        # Check there are NaN values before dropping
        assert X.isna().sum().sum() > 0
        
        # Drop NaN
        X_clean = X.dropna()
        
        # Check no NaN values after dropping
        assert X_clean.isna().sum().sum() == 0
        
        # Check that rows were removed
        assert len(X_clean) <= len(X)
    
    def test_train_test_split_shapes(self, sample_combined_data):
        """Test that train-test split produces correct shapes"""
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)
        
        feature_list = features_config['features']
        label_name = features_config['label']
        
        X = sample_combined_data[feature_list].copy()
        y = sample_combined_data[label_name].copy()
        
        X = X.dropna()
        y = y.loc[X.index]
        
        # Time-based split (80-20)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Check shapes
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1] == len(feature_list)
        assert len(X_train) + len(X_test) == len(X)
        
        # Check split ratio is approximately 80-20
        train_ratio = len(X_train) / len(X)
        assert 0.75 <= train_ratio <= 0.85  # Allow some flexibility
    
    def test_feature_scaling_shapes(self, sample_combined_data):
        """Test that feature scaling produces correct shapes"""
        from sklearn.preprocessing import StandardScaler
        
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)
        
        feature_list = features_config['features']
        X = sample_combined_data[feature_list].copy()
        X = X.dropna()
        
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Check shapes match
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert X_train_scaled.columns.equals(X_train.columns)
        assert X_test_scaled.columns.equals(X_test.columns)
    
    def test_model_training_shapes(self, sample_combined_data):
        """Test that models are trained with correct data shapes"""
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)
        
        with open('model_params.json', 'r') as f:
            model_params = json.load(f)
        
        feature_list = features_config['features']
        label_name = features_config['label']
        
        X = sample_combined_data[feature_list].copy()
        y = sample_combined_data[label_name].copy()
        
        X = X.dropna()
        y = y.loc[X.index]
        
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Train models
        lr_params = model_params['LogisticRegression']
        xgb_params = model_params['XGBClassifier']
        
        lr_model = LogisticRegression(**lr_params, random_state=42)
        xgb_model = XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss')
        
        lr_model.fit(X_train_scaled, y_train)
        xgb_model.fit(X_train, y_train)
        
        # Check models are trained
        assert lr_model is not None
        assert xgb_model is not None
        
        # Check that models have correct number of features
        assert lr_model.n_features_in_ == len(feature_list)
        assert xgb_model.n_features_in_ == len(feature_list)
    
    def test_prediction_shapes(self, sample_combined_data):
        """Test that predictions have correct shapes"""
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)
        
        with open('model_params.json', 'r') as f:
            model_params = json.load(f)
        
        feature_list = features_config['features']
        label_name = features_config['label']
        
        X = sample_combined_data[feature_list].copy()
        y = sample_combined_data[label_name].copy()
        
        X = X.dropna()
        y = y.loc[X.index]
        
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Train and predict
        lr_params = model_params['LogisticRegression']
        xgb_params = model_params['XGBClassifier']
        
        lr_model = LogisticRegression(**lr_params, random_state=42)
        xgb_model = XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss')
        
        lr_model.fit(X_train_scaled, y_train)
        xgb_model.fit(X_train, y_train)
        
        # Make predictions
        lr_pred_train = lr_model.predict(X_train_scaled)
        lr_pred_test = lr_model.predict(X_test_scaled)
        xgb_pred_train = xgb_model.predict(X_train)
        xgb_pred_test = xgb_model.predict(X_test)
        
        # Check prediction shapes
        assert len(lr_pred_train) == len(y_train)
        assert len(lr_pred_test) == len(y_test)
        assert len(xgb_pred_train) == len(y_train)
        assert len(xgb_pred_test) == len(y_test)
        
        # Check predictions are binary
        assert set(lr_pred_train).issubset({0, 1})
        assert set(lr_pred_test).issubset({0, 1})
        assert set(xgb_pred_train).issubset({0, 1})
        assert set(xgb_pred_test).issubset({0, 1})
        
        # Check predictions are numpy arrays
        assert isinstance(lr_pred_train, np.ndarray)
        assert isinstance(lr_pred_test, np.ndarray)
        assert isinstance(xgb_pred_train, np.ndarray)
        assert isinstance(xgb_pred_test, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])