from src.ml.training import AWSCostTrainer
import os

if __name__ == "__main__":
    # Absolute path to the CSV file
    csv_path = r'D:\Projects\Cost-Prediction-AWS(Iot Core Integrated)\data\raw\aws_billing_with_iot.csv'
    
    trainer = AWSCostTrainer({
        'epochs': 50,
        'batch_size': 32,
        'model_dir': '../models'  # You can also convert this to an absolute path if needed
    })

    trainer.train(csv_path)
