# src/services/iot_cost.py
from typing import List, Dict
from src.schemas import IoTRule, IoTPolicy, IoTOptimizationRecommendation
import numpy as np

class IoTCostService:
    # Cost constants (based on AWS pricing as of 2023)
    MESSAGE_BASE_COST = 0.0000005  # $0.50 per million messages
    RULE_BASE_COST = 0.0000015    # $1.50 per million rule evaluations
    ACTION_BASE_COST = 0.0000002  # $0.20 per million actions
    POLICY_EVAL_COST = 0.0000001  # $0.10 per million policy evaluations
    
    def __init__(self):
        # These could be loaded from a trained model in a real implementation
        self.rule_complexity_factors = {
            'simple': 1.0,    # Simple WHERE clauses
            'medium': 1.5,    # JOINs or moderate complexity
            'complex': 2.5    # Advanced SQL features
        }
        
        self.policy_complexity_factors = {
            'simple': 1.0,    # Single statement
            'medium': 1.8,    # Multiple statements
            'complex': 3.0     # Complex conditions
        }
    
    def predict_cost(self, message_volume: int, rules: List[IoTRule], 
                    policies: List[IoTPolicy]) -> Dict:
        # Calculate message cost
        message_cost = message_volume * self.MESSAGE_BASE_COST
        
        # Calculate rule execution cost
        rule_cost = 0
        for rule in rules:
            if rule.enabled:
                # Estimate rule complexity
                complexity = self._estimate_rule_complexity(rule.sql)
                evaluations = message_volume * complexity
                rule_cost += evaluations * self.RULE_BASE_COST
                
                # Add action costs
                rule_cost += len(rule.actions) * message_volume * self.ACTION_BASE_COST
        
        # Calculate policy evaluation cost
        policy_cost = 0
        for policy in policies:
            complexity = self._estimate_policy_complexity(policy)
            evaluations = message_volume * complexity
            policy_cost += evaluations * self.POLICY_EVAL_COST
            
        total_cost = message_cost + rule_cost + policy_cost
        
        return {
            "message_cost": message_cost,
            "rule_execution_cost": rule_cost,
            "policy_evaluation_cost": policy_cost,
            "total_cost": total_cost,
            "cost_breakdown": {
                "messages": message_cost / total_cost * 100,
                "rules": rule_cost / total_cost * 100,
                "policies": policy_cost / total_cost * 100
            }
        }
    
    def get_optimization_recommendations(self, message_volume: int, 
                                       rules: List[IoTRule], 
                                       policies: List[IoTPolicy]) -> List[IoTOptimizationRecommendation]:
        current_cost = self.predict_cost(message_volume, rules, policies)
        recommendations = []
        
        # Rule optimization recommendations
        enabled_rules = [r for r in rules if r.enabled]
        if len(enabled_rules) > 5:
            # Recommend combining rules
            potential_savings = 0.3 * current_cost['rule_execution_cost']
            recommendations.append(
                IoTOptimizationRecommendation(
                    area="rules",
                    current_cost=current_cost['rule_execution_cost'],
                    potential_savings=potential_savings,
                    recommendation="Combine multiple rules into fewer, more efficient rules",
                    implementation="Analyze rule SQL for common patterns and combine using CASE statements"
                )
            )
        
        # Policy optimization
        if len(policies) > 3:
            potential_savings = 0.25 * current_cost['policy_evaluation_cost']
            recommendations.append(
                IoTOptimizationRecommendation(
                    area="policies",
                    current_cost=current_cost['policy_evaluation_cost'],
                    potential_savings=potential_savings,
                    recommendation="Consolidate policies with similar statements",
                    implementation="Merge policies with overlapping permissions using policy variables"
                )
            )
        
        # Message processing optimization
        if message_volume > 1000000:  # More than 1 million messages
            potential_savings = 0.15 * current_cost['message_cost']
            recommendations.append(
                IoTOptimizationRecommendation(
                    area="message_processing",
                    current_cost=current_cost['message_cost'],
                    potential_savings=potential_savings,
                    recommendation="Implement message batching",
                    implementation="Use IoT BatchPutMessage API to reduce per-message overhead"
                )
            )
        
        return recommendations
    
    def _estimate_rule_complexity(self, sql: str) -> float:
        # Simplified complexity estimation
        sql_lower = sql.lower()
        
        if 'join' in sql_lower or 'group by' in sql_lower:
            return self.rule_complexity_factors['medium']
        elif 'case when' in sql_lower or 'regexp_match' in sql_lower:
            return self.rule_complexity_factors['complex']
        else:
            return self.rule_complexity_factors['simple']
    
    def _estimate_policy_complexity(self, policy: IoTPolicy) -> float:
        # Simplified policy complexity estimation
        if len(policy.statements) > 3:
            return self.policy_complexity_factors['complex']
        elif len(policy.statements) > 1:
            return self.policy_complexity_factors['medium']
        else:
            return self.policy_complexity_factors['simple']