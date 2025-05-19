# aws_services_cost_data_generator.py
import csv
import random
import time
from datetime import datetime

csv_file = 'aws_services_cost_data.csv'

header = [
    "timestamp",
    "region",
    "ec2_instance_hours",
    "ec2_instance_type",
    "ec2_data_transfer_gb",
    "s3_storage_gb",
    "s3_requests",
    "lambda_invocations",
    "lambda_duration_ms",
    "cloudwatch_logs_gb",
    "dynamodb_read_units",
    "dynamodb_write_units",
    "api_gateway_requests",
    "estimated_cost_usd"
]

ec2_types = {
    "t2.micro": 0.0116,
    "t3.medium": 0.0416,
    "m5.large": 0.096
}

regions = ["us-east-1", "us-west-2", "eu-central-1"]

try:
    with open(csv_file, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
except FileExistsError:
    pass

while True:
    ec2_type = random.choice(list(ec2_types.keys()))
    ec2_hours = random.uniform(0, 24)
    s3_gb = random.uniform(10, 1000)
    lambda_inv = random.randint(100, 10000)
    lambda_dur = random.randint(100, 1000)
    api_req = random.randint(1000, 50000)

    # Simulate cost
    cost = (
        ec2_types[ec2_type] * ec2_hours +
        s3_gb * 0.023 +
        lambda_inv * lambda_dur * 1e-7 +
        api_req * 0.0000035 +
        random.uniform(0.1, 2.0)  # misc (CloudWatch, DynamoDB, etc.)
    )

    row = [
        datetime.utcnow().isoformat(),
        random.choice(regions),
        round(ec2_hours, 2),
        ec2_type,
        round(random.uniform(1, 100), 2),  # ec2_data_transfer_gb
        round(s3_gb, 2),
        random.randint(100, 10000),        # s3_requests
        lambda_inv,
        lambda_dur,
        round(random.uniform(0.1, 10.0), 2),  # cloudwatch_logs_gb
        random.randint(100, 1000),  # dynamodb_read_units
        random.randint(100, 1000),  # dynamodb_write_units
        api_req,
        round(cost, 4)
    ]

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("AWS Cost Data Logged:", row)
    time.sleep(5)
