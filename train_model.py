'''
Trains and evaluates ML models
'''

import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TrainModel:
    def __init__(self, df_combined = None):
        self.df_combined = df_combined
        self.lr_model = None
        self.xgb_model = None
        self.lr_pred_train = None
        self.lr_pred_test = None
        self.xgb_pred_train = None
        self.xgb_pred_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None  # Add scaler to store it

    def train_model(self):

        # Load configuration files
        with open('features_config.json', 'r') as f:
            features_config = json.load(f)

        with open('model_params.json', 'r') as f:
            model_params = json.load(f)

        df_combined = pd.read_csv('df_combined.csv')

        # Get feature list and label name from config
        feature_list = features_config['features']
        label_name = features_config['label']

        print(f"Features to use: {feature_list}")
        print(f"Label: {label_name}")

        df_combined['date'] = pd.to_datetime(df_combined['date'])
        df_combined = df_combined.sort_values('date').reset_index(drop=True)

        # Select features and target
        X = df_combined[feature_list].copy()
        y = df_combined[label_name].copy()

        # Drop rows with missing values (from lag features and rolling windows)
        print(f"\nBefore dropping NaN: {X.shape}")
        X = X.dropna()
        y = y.loc[X.index]
        print(f"After dropping NaN: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")

        # Time-based train-test split (80-20 split)
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx].copy()  # Assign to self
        self.X_test = X.iloc[split_idx:].copy()   # Assign to self
        self.y_train = y.iloc[:split_idx].copy()  # Assign to self
        self.y_test = y.iloc[split_idx:].copy()   # Assign to self

        print(f"Train set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

        # Scale features (important for LogisticRegression)
        self.scaler = StandardScaler()  # Store scaler
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        # Initialize models with hyperparameters from config
        lr_params = model_params['LogisticRegression']
        xgb_params = model_params['XGBClassifier']

        self.lr_model = LogisticRegression(**lr_params, random_state=42)  # Assign to self
        self.xgb_model = XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss')  # Assign to self

        # Train models
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        self.lr_model.fit(self.X_train_scaled, self.y_train)

        print("Training XGBoost...")
        self.xgb_model.fit(self.X_train, self.y_train)

        # Make predictions
        self.lr_pred_train = self.lr_model.predict(self.X_train_scaled)
        self.lr_pred_test = self.lr_model.predict(self.X_test_scaled)

        self.xgb_pred_train = self.xgb_model.predict(self.X_train)
        self.xgb_pred_test = self.xgb_model.predict(self.X_test)

    def evaluate_model(self): 

        # Evaluate on test set
        print("\n" + "="*50)
        print("TEST SET PERFORMANCE")
        print("="*50)

        models = {
            'Logistic Regression': (self.lr_pred_test, self.y_test),
            'XGBoost': (self.xgb_pred_test, self.y_test)
        }

        results = {}

        for model_name, (pred, true) in models.items():
            accuracy = accuracy_score(true, pred)
            precision = precision_score(true, pred, zero_division=0)
            recall = recall_score(true, pred, zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")

        # Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for idx, (model_name, (pred, true)) in enumerate(models.items()):
            cm = confusion_matrix(true, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

        # Cross-Validation (Time Series Split to avoid data leakage)
        print("\n" + "="*50)
        print("CROSS-VALIDATION RESULTS")
        print("="*50)

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)

        # Cross-validation for Logistic Regression
        lr_cv_scores = cross_val_score(
            self.lr_model, self.X_train_scaled, self.y_train,
            cv=tscv, scoring='accuracy', n_jobs=-1
        )

        # Cross-validation for XGBoost
        xgb_cv_scores = cross_val_score(
            self.xgb_model, self.X_train, self.y_train,
            cv=tscv, scoring='accuracy', n_jobs=-1
        )

        print(f"\nLogistic Regression CV Accuracy:")
        print(f"  Mean: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")
        print(f"  Scores: {lr_cv_scores}")

        print(f"\nXGBoost CV Accuracy:")
        print(f"  Mean: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std() * 2:.4f})")
        print(f"  Scores: {xgb_cv_scores}")

        # Detailed classification reports
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*50)

        print("\nLogistic Regression:")
        print(classification_report(self.y_test, self.lr_pred_test, zero_division=0))

        print("\nXGBoost:")
        print(classification_report(self.y_test, self.xgb_pred_test, zero_division=0))

        # Feature importance for XGBoost
        print("\n" + "="*50)
        print("XGBOOST FEATURE IMPORTANCE")
        print("="*50)

        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(feature_importance)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('XGBoost Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

    def save_data(self):
        # Create data directory if it doesn't exist
        import os
        os.makedirs('data', exist_ok=True)
        
        self.df_combined.to_csv('data/df_combined.csv', index=False) if self.df_combined is not None else None
        pd.Series(self.lr_pred_train).to_csv('data/lr_pred_train.csv', index=False) if self.lr_pred_train is not None else None
        pd.Series(self.lr_pred_test).to_csv('data/lr_pred_test.csv', index=False) if self.lr_pred_test is not None else None
        pd.Series(self.xgb_pred_train).to_csv('data/xgb_pred_train.csv', index=False) if self.xgb_pred_train is not None else None
        pd.Series(self.xgb_pred_test).to_csv('data/xgb_pred_test.csv', index=False) if self.xgb_pred_test is not None else None
        self.y_test.to_csv('data/y_test.csv', index=False) if self.y_test is not None else None
        self.X_train.to_csv('data/X_train.csv', index=False) if self.X_train is not None else None
        self.X_test.to_csv('data/X_test.csv', index=False) if self.X_test is not None else None
        self.X_train_scaled.to_csv('data/X_train_scaled.csv', index=False) if self.X_train_scaled is not None else None
        self.X_test_scaled.to_csv('data/X_test_scaled.csv', index=False) if self.X_test_scaled is not None else None
        self.y_train.to_csv('data/y_train.csv', index=False) if self.y_train is not None else None