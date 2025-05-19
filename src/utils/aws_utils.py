def generate_cost_breakdown(input_data):
    """Generate AWS cost breakdown based on new features"""
    breakdown = []
    
    # EC2 Costs
    ec2_cost = (input_data['ec2_instance_hours'] * 0.05 +  # Instance hours
                input_data['ec2_data_transfer_gb'] * 0.09)  # Data transfer
    breakdown.append({
        "service": "EC2",
        "usage": f"{input_data['ec2_instance_hours']} hours, {input_data['ec2_data_transfer_gb']}GB",
        "cost": ec2_cost,
        "explanation": "Based on instance hours and data transfer"
    })
    
    # S3 Costs
    s3_cost = (input_data['s3_storage_gb'] * 0.023 +  # Storage
               input_data['s3_requests'] * 0.0004)     # Requests
    breakdown.append({
        "service": "S3",
        "usage": f"{input_data['s3_storage_gb']}GB, {input_data['s3_requests']} requests",
        "cost": s3_cost,
        "explanation": "Based on storage and request counts"
    })
    
    # Add other services similarly
    return breakdown

def generate_recommendations(input_data):
    """Generate AWS optimization recommendations"""
    recommendations = []
    
    # EC2 recommendations
    if input_data['ec2_instance_hours'] > 720:  # > 30 days
        recommendations.append({
            "current_setup": "On-demand instances",
            "recommended_setup": "Reserved Instances (1-year)",
            "potential_savings": input_data['ec2_instance_hours'] * 0.02,
            "implementation_complexity": "Medium"
        })
    
    # S3 recommendations
    if input_data['s3_storage_gb'] > 500 and input_data['s3_requests'] < 1000:
        recommendations.append({
            "current_setup": "Standard storage",
            "recommended_setup": "Infrequent Access storage class",
            "potential_savings": input_data['s3_storage_gb'] * 0.01,
            "implementation_complexity": "Low"
        })
    
    return recommendations