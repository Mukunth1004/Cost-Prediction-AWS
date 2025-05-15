import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_dummy_data(days=180):
    """Generate realistic dummy AWS billing data with IoT Core metrics"""
    dates = pd.date_range(end=datetime.now(), periods=days).strftime('%Y-%m-%d')
    services = [
        'AmazonEC2', 'AmazonS3', 'AmazonRDS', 'AWSIoT', 
        'AmazonLambda', 'AmazonCloudFront', 'AmazonAPIGateway'
    ]
    
    # Base costs per service (simulated monthly averages in USD)
    service_base_costs = {
        'AmazonEC2': 50,
        'AmazonS3': 5,
        'AmazonRDS': 30,
        'AWSIoT': 20,
        'AmazonLambda': 15,
        'AmazonCloudFront': 25,
        'AmazonAPIGateway': 10
    }
    
    records = []
    for date in dates:
        daily_services_cost = {}
        
        for service in services:
            # Simulate daily variation (±30%)
            base_cost = service_base_costs[service] / 30  # Convert monthly to daily
            cost = base_cost * (0.7 + 0.6 * random.random())

            # IoT-specific metrics (only for AWSIoT)
            iot_things = random.randint(10, 100) if service == 'AWSIoT' else 0
            iot_rules = random.randint(1, 10) if service == 'AWSIoT' else 0
            iot_policies = random.randint(1, 5) if service == 'AWSIoT' else 0

            # Add IoT cost components
            if service == 'AWSIoT':
                cost += (iot_things * 0.001) + (iot_rules * 0.01) + (iot_policies * 0.005)

            daily_services_cost[service] = cost

            records.append({
                'Date': date,
                'Service': service,
                'Cost': round(cost, 4),
                'Currency': 'USD',
                'IoT_Things': iot_things,
                'IoT_Rules': iot_rules,
                'IoT_Policies': iot_policies,
                'Is_IoT_Core': 1 if service == 'AWSIoT' else 0
            })

    df = pd.DataFrame(records)

    # Add weekly seasonality
    df['Date'] = pd.to_datetime(df['Date'])
    df['Cost'] = df.apply(lambda row: row['Cost'] * (1.2 if row['Date'].dayofweek in [0, 4] else 1), axis=1)

    # Save to CSV
    save_path = r'D:\Projects\Cost-Prediction-AWS(Iot Core Integrated)\data\raw\dummy_aws_billing.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Generated {len(df)} records in {save_path}")
    return df

if __name__ == "__main__":
    generate_dummy_data(days=180)
