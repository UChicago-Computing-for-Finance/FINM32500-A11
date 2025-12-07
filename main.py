

from feature_engineering import FeatureEngineering
from train_model import TrainModel
from signal_generator import SignalGenerator

if __name__ == "__main__":
    
    feature_engineering = FeatureEngineering()
    feature_engineering.create_features()

    train_model = TrainModel()
    train_model.train_model()
    train_model.evaluate_model()
    train_model.save_data()

    signal_generator = SignalGenerator()
    signal_generator.run_back()
