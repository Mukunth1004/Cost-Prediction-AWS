import pandas as pd
from src.ml.data_preprocessing import DataPreprocessor
from src.ml.feature_engineering import FeatureEngineer
from src.ml.models import HybridModel
from src.ml.training import AWSCostTrainer

def main():
    # Set correct absolute path
    csv_path = r'D:\Projects\Cost-Prediction-AWS(Iot Core Integrated)\data\raw\dummy_aws_billing.csv'
    
    # 1. Load dummy data
    df = pd.read_csv(csv_path)
    
    # 2. Preprocess data
    preprocessor = DataPreprocessor()
   
    
    # 3. Feature engineering (time-series features)
    feature_engineer = FeatureEngineer()
   
    
    # 4. Train model
    trainer = AWSCostTrainer({
        'epochs': 30,  # Reduced for dummy data
        'batch_size': 32,
        'model_dir': r'D:\Projects\Cost-Prediction-AWS(Iot Core Integrated)\models'
    })
    mae, rmse = trainer.train(csv_path)
    
    print(f"\nTraining complete! MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # 6. (Optional) Verify preprocessing
    # To see the fully processed data that was used for training:
    df_loaded = trainer.load_data(csv_path)
    df_preprocessed = trainer.preprocessor.preprocess_aws_data(df_loaded)
    df_with_features = trainer.feature_engineer.create_time_series_features(df_preprocessed)
    print("\nSample of fully processed data:")
    print(df_with_features.head())

if __name__ == "__main__":
    main()
