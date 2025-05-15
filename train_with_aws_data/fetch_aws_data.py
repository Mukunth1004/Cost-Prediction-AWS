import boto3
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

# Load AWS credentials
load_dotenv()

class AWSBillingFetcher:
    def __init__(self):
        self.ce_client = boto3.client(
            'ce',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.iot_client = boto3.client('iot')

    def get_cost_data(self, days=90):
        """Fetch cost and IoT Core data"""
        # 1. Get standard AWS costs
        cost_df = self._get_standard_costs(days)
        
        # 2. Get IoT Core-specific data
        iot_df = self._get_iot_core_metrics(days)
        
        # Merge datasets
        return pd.concat([cost_df, iot_df], ignore_index=True)

    def _get_standard_costs(self, days):
        """Fetch regular AWS service costs"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            return self._process_cost_response(response)
        except Exception as e:
            print(f"Error fetching cost data: {e}")
            return pd.DataFrame()

    def _get_iot_core_metrics(self, days):
        """Fetch IoT Core-specific metrics"""
        try:
            # Get IoT Things count
            things = self.iot_client.list_things()['things']
            
            # Get IoT Rules
            rules = self.iot_client.list_topic_rules()['rules']
            
            # Get IoT Policies
            policies = self.iot_client.list_policies()['policies']
            
            # Create IoT-specific cost estimation
            return self._estimate_iot_costs(days, len(things), len(rules), len(policies))
        except Exception as e:
            print(f"Error fetching IoT data: {e}")
            return pd.DataFrame()

    def _process_cost_response(self, response):
        """Process standard cost data"""
        records = []
        for day in response['ResultsByTime']:
            for group in day['Groups']:
                records.append({
                    'Date': day['TimePeriod']['Start'],
                    'Service': group['Keys'][0],
                    'Cost': float(group['Metrics']['UnblendedCost']['Amount']),
                    'Currency': group['Metrics']['UnblendedCost']['Unit'],
                    'IoT_Things': 0,  # Placeholder
                    'IoT_Rules': 0,
                    'IoT_Policies': 0,
                    'Is_IoT_Core': 1 if group['Keys'][0] == 'AWSIoT' else 0
                })
        return pd.DataFrame(records)

    def _estimate_iot_costs(self, days, thing_count, rule_count, policy_count):
        """Estimate IoT Core costs based on usage patterns"""
        dates = pd.date_range(end=datetime.now(), periods=days).strftime('%Y-%m-%d')
        
        # Base costs (example values - adjust based on your AWS pricing)
        BASE_THING_COST = 0.0001  # $ per thing per day
        BASE_RULE_COST = 0.001    # $ per rule per day
        BASE_POLICY_COST = 0.0005 # $ per policy per day
        
        records = []
        for date in dates:
            # Simulate daily variation (±20%)
            variation = 1 + (0.2 * (hash(date) % 100 - 50) / 100  )
            
            records.append({
                'Date': date,
                'Service': 'AWSIoT',
                'Cost': (
                    (thing_count * BASE_THING_COST) +
                    (rule_count * BASE_RULE_COST) +
                    (policy_count * BASE_POLICY_COST)
                ) * variation,
                'Currency': 'USD',
                'IoT_Things': thing_count,
                'IoT_Rules': rule_count,
                'IoT_Policies': policy_count,
                'Is_IoT_Core': 1
            })
        
        return pd.DataFrame(records)

if __name__ == "__main__":
    fetcher = AWSBillingFetcher()
    df = fetcher.get_cost_data(days=180)  # Last 6 months
    
    if not df.empty:
        # Save to CSV
        os.makedirs('../data/raw', exist_ok=True)
        csv_path = '../data/raw/aws_billing_with_iot.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"Data saved to {csv_path}")
        print("\nSample IoT Core data:")
        print(df[df['Is_IoT_Core'] == 1].head())
        print("\nSample other services:")
        print(df[df['Is_IoT_Core'] == 0].head())