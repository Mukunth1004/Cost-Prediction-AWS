# src/utils/aws_client.py
import boto3
from datetime import datetime, timedelta
import pandas as pd

class AWSClient:
    def __init__(self):
        self.client = boto3.client('ce')
        
    def get_cost_data(self, days=90):
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        response = self.client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        # Process response into DataFrame
        data = []
        for day in response['ResultsByTime']:
            for group in day['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                data.append({
                    'Date': day['TimePeriod']['Start'],
                    'Service': service,
                    'Cost': cost
                })
                
        return pd.DataFrame(data)