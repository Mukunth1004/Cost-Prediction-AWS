import csv
import random
from datetime import datetime

csv_file = 'aws_iot_core_cost_data.csv'

header = [
    "timestamp",
    "thing_name",
    "thing_type",
    "region",
    "attached_policies",   # number of policies attached
    "policy_names",        # comma-separated names
    "attached_rules",      # number of rules attached
    "rule_names",          # comma-separated names
    "shadow_updates_per_day",
    "mqtt_messages_per_day",
    "http_requests_per_day",
    "device_connected_hours",
    "connection_type",
    "iot_data_transfer_mb",
    "estimated_cost_usd"
]

things = ["temperature_sensor_01", "motion_detector_02", "weather_station_03"]
thing_types = ["sensor", "actuator", "controller"]
regions = ["us-east-1", "ap-south-1", "eu-west-1"]
connections = ["mqtt", "http"]

rule_names = [
    "StoreToDynamoDB",
    "TriggerLambdaAlert",
    "ForwardToKinesis",
    "StoreToS3",
    "ShadowUpdateMonitor",
    "SendToSNS",
    "ArchiveToTimestream",
    "RepublishToTopic",
    "IoTAnalyticsPipeline",
    "InvokeStepFunction",
    "ElasticSearchIndex",
    "RouteToGreengrass",
    "LogToCloudWatch",
    "DetectAnomalyAndAlert",
    "MirrorToBackupRegion"
]

policy_names = [
    "IoT_Policy_AllowAll",
    "IoT_Restricted_Policy",
    "IoT_Custom_Policy_ReadWrite",
    "IoT_Limited_Policy_LogsOnly",
    "IoT_Policy_ConnectOnly",
    "IoT_Policy_TelemetryDataOnly",
    "IoT_Policy_ShadowAccess",
    "IoT_Policy_AdminAccess",
    "IoT_Policy_StorageOnly",
    "IoT_Policy_LambdaTrigger",
    "IoT_Policy_KinesisStreaming",
    "IoT_Policy_Monitoring",
    "IoT_Policy_FirmwareUpdate",
    "IoT_Policy_GreengrassLocal",
    "IoT_Policy_AnalyticsIngest"
]

rows = []
for _ in range(50):
    mqtt_msgs = random.randint(500, 20000)
    http_reqs = random.randint(0, 5000)
    shadow_updates = random.randint(0, 1000)
    connected_hours = random.uniform(1, 24)
    data_transfer = random.uniform(1, 100)

    # Cost estimation formula (dummy approximation)
    cost = (
        mqtt_msgs * 0.00000125 +   # MQTT messages
        http_reqs * 0.000004 +
        shadow_updates * 0.000002 +
        data_transfer * 0.12 +     # Data transfer MB
        connected_hours * 0.005    # Connected hours cost factor
    )

    attached_policy_count = random.randint(1, 3)
    attached_policies = random.sample(policy_names, attached_policy_count)

    attached_rule_count = random.randint(1, 2)
    attached_rules = random.sample(rule_names, attached_rule_count)

    row = [
        datetime.utcnow().isoformat(),
        random.choice(things),
        random.choice(thing_types),
        random.choice(regions),
        attached_policy_count,
        ",".join(attached_policies),
        attached_rule_count,
        ",".join(attached_rules),
        shadow_updates,
        mqtt_msgs,
        http_reqs,
        round(connected_hours, 2),
        random.choice(connections),
        round(data_transfer, 2),
        round(cost, 4)
    ]

    rows.append(row)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Generated {len(rows)} rows and saved to {csv_file}")
