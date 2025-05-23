# src/services/optimization_service.py

from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OptimizationService:
    @staticmethod
    def optimize_aws_resources(cost_breakdown: Dict, current_config: Dict) -> List[str]:
        """Generate AWS resource optimization suggestions"""
        suggestions = []

        if cost_breakdown.get('ec2', {}).get('percentage', 0) > 40:
            suggestions.append(
                "EC2 accounts for more than 40% of your costs. Consider:\n"
                "1. Right-sizing EC2 instances\n"
                "2. Using Spot Instances for non-critical workloads\n"
                "3. Purchasing Reserved Instances for predictable usage"
            )

        if cost_breakdown.get('s3', {}).get('percentage', 0) > 20:
            suggestions.append(
                "S3 accounts for more than 20% of your costs. Consider:\n"
                "1. Moving rarely accessed data to S3 Infrequent Access or Glacier\n"
                "2. Enabling lifecycle policies\n"
                "3. Compressing and deduplicating files before upload"
            )

        if cost_breakdown.get('lambda', {}).get('percentage', 0) > 10:
            suggestions.append(
                "Lambda accounts for more than 10% of your costs. Consider:\n"
                "1. Optimizing memory allocation and code\n"
                "2. Reducing invocation frequency\n"
                "3. Using Provisioned Concurrency for stable workloads"
            )

        if cost_breakdown.get('cloudwatch', {}).get('percentage', 0) > 10:
            suggestions.append(
                "CloudWatch logs cost is significant. Consider:\n"
                "1. Lowering log retention period\n"
                "2. Filtering and exporting only necessary logs\n"
                "3. Archiving logs to S3"
            )

        if cost_breakdown.get('dynamodb', {}).get('percentage', 0) > 10:
            suggestions.append(
                "DynamoDB usage is high. Consider:\n"
                "1. Switching to on-demand mode for unpredictable workloads\n"
                "2. Using DAX caching\n"
                "3. Reducing read/write throughput if underutilized"
            )

        if cost_breakdown.get('api_gateway', {}).get('percentage', 0) > 10:
            suggestions.append(
                "API Gateway costs are noticeable. Consider:\n"
                "1. Caching responses\n"
                "2. Reducing request rates with throttling\n"
                "3. Consolidating APIs or using ALB for some workloads"
            )

        return suggestions

    @staticmethod
    def generate_overall_optimization_suggestions(all_breakdowns: List[Dict]) -> List[str]:
        """Aggregate and generate high-level cost-saving recommendations"""
        total_cost = 0
        service_costs = {}

        for breakdown in all_breakdowns:
            for service, data in breakdown.items():
                cost = data.get("cost", 0)
                total_cost += cost
                service_costs[service] = service_costs.get(service, 0) + cost

        if total_cost == 0:
            return []

        overall_suggestions = []

        for service, cost in service_costs.items():
            percentage = (cost / total_cost) * 100

            if service == 'ec2' and percentage > 40:
                overall_suggestions.append("EC2 is the primary cost driver across all entries. Consider instance optimization.")
            elif service == 's3' and percentage > 25:
                overall_suggestions.append("S3 storage is a significant cost. Review object lifecycles and storage classes.")
            elif service == 'lambda' and percentage > 15:
                overall_suggestions.append("High Lambda usage overall. Optimize function size, frequency, and concurrency.")
            elif service == 'dynamodb' and percentage > 15:
                overall_suggestions.append("DynamoDB is contributing significantly to cost. Consider usage patterns and provisioned throughput.")
            elif service == 'cloudwatch' and percentage > 10:
                overall_suggestions.append("CloudWatch logs are consuming budget. Manage retention and log level.")
            elif service == 'api_gateway' and percentage > 10:
                overall_suggestions.append("API Gateway costs are high. Reduce redundant requests or cache results.")

        return overall_suggestions

    
    @staticmethod
    def optimize_iot_resources(cost_breakdown: Dict, current_config: Dict) -> List[str]:
        """Generate IoT Core resource optimization suggestions"""
        suggestions = []
        
        # Rules Engine Optimization
        if cost_breakdown.get('rules_engine', {}).get('percentage', 0) > 30:
            suggestions.append(
                "Rules Engine accounts for more than 30% of your costs. Consider: "
                "1. Combining multiple rules into fewer, more efficient rules\n"
                "2. Using more efficient SQL queries in your rules\n"
                "3. Reducing rule evaluation frequency where possible"
            )
        
        # Message Optimization
        message_cost = cost_breakdown.get('mqtt_messages', {}).get('cost', 0) + \
                     cost_breakdown.get('http_requests', {}).get('cost', 0)
        total_cost = sum(item.get('cost', 0) for item in cost_breakdown.values())
        
        if total_cost > 0 and (message_cost / total_cost) > 0.4:
            suggestions.append(
                "Messaging accounts for more than 40% of your costs. Consider: "
                "1. Batching device messages\n"
                "2. Using MQTT instead of HTTP where possible\n"
                "3. Reducing message frequency where possible"
            )
        
        # Shadow Optimization
        if cost_breakdown.get('device_shadow', {}).get('percentage', 0) > 20:
            suggestions.append(
                "Device Shadows account for more than 20% of your costs. Consider: "
                "1. Reducing shadow update frequency\n"
                "2. Using smaller shadow documents\n"
                "3. Only updating shadows when significant changes occur"
            )
        
        return suggestions