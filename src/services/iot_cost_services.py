import numpy as np
import pandas as pd
import joblib
from fastapi import HTTPException  # Add at top of file
from src.models.iot_model import IoTModel
from src.data_processing.iot_data_preprocessor import IoTDataPreprocessor
from src.schemas.iot_schema import IoTInput
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IoTCostService:
    def __init__(self):
        self.model = IoTModel()
        self.preprocessor = IoTDataPreprocessor()
        self.is_trained = False

    def train_model(self):
        """Train the IoT cost prediction model"""
        try:
            # Load and preprocess data
            df = self.preprocessor.load_data(config.IOT_DATA_PATH)
            if df is None:
                raise ValueError("No IoT data loaded")
            X, y = self.preprocessor.prepare_data(df, is_training=True)
            if X is None or y is None:
                raise ValueError("IoT data preparation failed")

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

            # Train model
            result = self.model.train(X_train, y_train, X_val, y_val)
            self.is_trained = True
            self.model.save_model()

            if result['status'] != 'success':
                raise ValueError(result['message'])

            return {
                'status': 'success',
                'metrics': result.get('metrics', {}),
                'feature_importances': result.get('feature_importances', [])
            }

        except Exception as e:
            logger.error(f"IoT training failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def load_model(self):
        """Load the trained model"""
        try:
            self.model.load_model()
            self.scaler = joblib.load(f"{config.SCALER_SAVE_PATH}/iot_scaler.pkl")
            self.is_trained = True
            logger.info("IoT cost prediction model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_trained = False
            return False

    def predict_cost(self, input_data: IoTInput):
        """Handle IoT predictions with proper input shape"""
        try:
            if not self.is_trained and not self.load_model():
                raise HTTPException(status_code=400, detail="Model not loaded")

            items = input_data.data
            predictions = []
            for item in items:
                # Convert Pydantic model to dict
                item_dict = item.dict(by_alias=True)

                # Ensure required fields exist with defaults
                item_dict.setdefault('iot_data_transfer_mb', 0.0)

                # Preprocess the input
                preprocessed_input = self.preprocessor.load_data_from_dict(item_dict)

                # Ensure proper shape (1, n_features) for single prediction
                if len(preprocessed_input.shape) == 1:
                    preprocessed_input = preprocessed_input.reshape(1, -1)

                # Make prediction
                pred = float(self.model.predict(preprocessed_input)[0])

                predictions.append({
                    'thing_name': item.thing_name,
                    'predicted_cost': pred,
                    'cost_breakdown': self._generate_cost_breakdown(item_dict),
                    'optimization_suggestions': self._generate_optimization_suggestions(item_dict, pred)
                })

            return {
                'predictions': predictions,
                'overall_optimization_suggestions': self._generate_overall_suggestions(
                    [item.dict(by_alias=True) for item in items],
                    [p['predicted_cost'] for p in predictions]
                ),
                'total_predicted_cost': sum(p['predicted_cost'] for p in predictions)
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    def _generate_cost_breakdown(self, item):
        """Generate detailed cost breakdown with explanations"""
        breakdown = {
            'rules': {
                'cost': item.get('attached_rules', 0) * 0.5,
                'explanation': f"Each rule costs $0.50/month. You have {item.get('attached_rules', 0)} rules attached.",
                'optimization': "Combine similar rules to reduce count"
            },
            'messaging': {
                'cost': item.get('mqtt_messages_per_day', 0) * 0.0001,
                'explanation': f"MQTT messages cost $0.0001 each. You send {item.get('mqtt_messages_per_day', 0)} messages/day.",
                'optimization': "Batch messages to reduce frequency"
            },
            'shadow': {
                'cost': item.get('shadow_updates_per_day', 0) * 0.00001,
                'explanation': f"Shadow updates cost $0.00001 each. You perform {item.get('shadow_updates_per_day', 0)} updates/day.",
                'optimization': "Reduce update frequency or batch updates"
            },
            'data_transfer': {
                'cost': item.get('iot_data_transfer_mb', 0) * 0.01,
                'explanation': f"Data transfer costs $0.01/MB. You transfer {item.get('iot_data_transfer_mb', 0)} MB/day.",
                'optimization': "Compress data before transfer"
            },
            'policies': {
                'cost': item.get('attached_policies', 0) * 0.3,
                'explanation': f"Each policy costs $0.30/month. You have {item.get('attached_policies', 0)} policies attached.",
                'optimization': "Consolidate policies with similar permissions"
            }
        }

        # Calculate total cost
        total = sum(v['cost'] for v in breakdown.values())

        # Add percentages
        for key in breakdown:
            breakdown[key]['percentage'] = (breakdown[key]['cost'] / total * 100) if total > 0 else 0

        return breakdown

    def _generate_optimization_suggestions(self, item, predicted_cost):
        """Generate detailed optimization suggestions"""
        suggestions = []

        # Rules optimization
        if item.get('attached_rules', 0) > 3:
            suggestions.append(
                f"Rules Cost: ${item.get('attached_rules', 0) * 0.5:.2f}/month - "
                f"You have {item.get('attached_rules', 0)} rules. "
                "Optimization: Combine similar rules to reduce count."
            )

        # Messaging optimization
        if item.get('mqtt_messages_per_day', 0) > 5000:
            suggestions.append(
                f"Messaging Cost: ${item.get('mqtt_messages_per_day', 0) * 0.0001 * 30:.2f}/month - "
                f"You send {item.get('mqtt_messages_per_day', 0)} messages/day. "
                "Optimization: Batch messages to reduce frequency."
            )

        # Shadow optimization
        if item.get('shadow_updates_per_day', 0) > 1000:
            suggestions.append(
                f"Shadow Cost: ${item.get('shadow_updates_per_day', 0) * 0.00001 * 30:.2f}/month - "
                f"You perform {item.get('shadow_updates_per_day', 0)} updates/day. "
                "Optimization: Reduce update frequency or batch updates."
            )

        # Data transfer optimization
        if item.get('iot_data_transfer_mb', 0) > 100:
            suggestions.append(
                f"Data Transfer Cost: ${item.get('iot_data_transfer_mb', 0) * 0.01 * 30:.2f}/month - "
                f"You transfer {item.get('iot_data_transfer_mb', 0)} MB/day. "
                "Optimization: Compress data before transfer."
            )

        # Policies optimization
        if item.get('attached_policies', 0) > 2:
            suggestions.append(
                f"Policies Cost: ${item.get('attached_policies', 0) * 0.3:.2f}/month - "
                f"You have {item.get('attached_policies', 0)} policies. "
                "Optimization: Consolidate policies with similar permissions."
            )

        return suggestions

    def _generate_overall_suggestions(self, items, predictions):
        """Generate high-level optimization suggestions"""
        if not items:
            return []

        # Calculate totals across all items
        total_cost = sum(predictions)
        total_rules = sum(item.get('attached_rules', 0) for item in items)
        total_messages = sum(item.get('mqtt_messages_per_day', 0) for item in items)
        total_shadow = sum(item.get('shadow_updates_per_day', 0) for item in items)
        total_data = sum(item.get('iot_data_transfer_mb', 0) for item in items)
        total_policies = sum(item.get('attached_policies', 0) for item in items)

        suggestions = []

        if total_rules / len(items) > 2:
            suggestions.append(
                f"Average rules per device: {total_rules / len(items):.1f}. "
                "Consider reviewing and combining rules across your IoT fleet."
            )

        if total_messages / len(items) > 3000:
            suggestions.append(
                f"Average messages per device: {total_messages / len(items):.1f}/day. "
                "Implement message batching across your IoT solution."
            )

        if total_shadow / len(items) > 500:
            suggestions.append(
                f"Average shadow updates per device: {total_shadow / len(items):.1f}/day. "
                "Review shadow update frequency across all devices."
            )

        if total_data / len(items) > 50:
            suggestions.append(
                f"Average data transfer per device: {total_data / len(items):.1f} MB/day. "
                "Implement data compression across your IoT solution."
            )

        if total_policies / len(items) > 1.5:
            suggestions.append(
                f"Average policies per device: {total_policies / len(items):.1f}. "
                "Standardize policies across similar devices."
            )

        return suggestions
