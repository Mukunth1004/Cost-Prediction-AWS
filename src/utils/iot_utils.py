def calculate_rule_costs(rules_list):
    """Calculate costs for specific IoT rules"""
    rule_costs = {
        'basic_rule': 0.05,
        'sql_rule': 0.10,
        'lambda_rule': 0.15,
        'republish_rule': 0.08
    }
    
    costs = []
    for rule in rules_list:
        rule_type = rule.split('_')[0] + '_rule'  # Simple extraction
        cost = rule_costs.get(rule_type, 0.07)
        costs.append({
            "rule_name": rule,
            "type": rule_type,
            "cost": cost
        })
    
    return costs

def calculate_policy_costs(policies_list):
    """Calculate costs for specific IoT policies"""
    policy_costs = {
        'basic_policy': 0.03,
        'certificate_policy': 0.05,
        'advanced_policy': 0.08
    }
    
    costs = []
    for policy in policies_list:
        policy_type = policy.split('_')[0] + '_policy'  # Simple extraction
        cost = policy_costs.get(policy_type, 0.04)
        costs.append({
            "policy_name": policy,
            "type": policy_type,
            "cost": cost
        })
    
    return costs

def generate_iot_optimizations(rule_costs, policy_costs, message_count):
    """Generate specific IoT optimizations"""
    optimizations = []
    
    # Rule optimizations
    expensive_rules = [r for r in rule_costs if r['cost'] > 0.10]
    if expensive_rules:
        optimizations.append({
            "current": f"Expensive rules: {', '.join([r['rule_name'] for r in expensive_rules])}",
            "recommendation": "Replace with basic rules where possible",
            "savings": sum(r['cost'] - 0.05 for r in expensive_rules),
            "complexity": "Medium"
        })
    
    # Policy optimizations
    if len(policy_costs) > 3:
        optimizations.append({
            "current": f"{len(policy_costs)} policies attached",
            "recommendation": "Consolidate policies to 2-3 comprehensive policies",
            "savings": (len(policy_costs) - 2) * 0.03,
            "complexity": "High"
        })
    
    # Message optimizations
    if message_count > 5000:
        optimizations.append({
            "current": f"{message_count} individual messages",
            "recommendation": "Implement message batching",
            "savings": message_count * 0.00003,
            "complexity": "Low"
        })
    
    return optimizations