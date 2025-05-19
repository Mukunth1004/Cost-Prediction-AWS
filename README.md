# 🚀 AWS-IoT Cost Prediction System

A powerful machine learning-based system to predict and optimize costs for AWS services and AWS IoT Core usage.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🔍 Features

- 🔮 **AWS Cost Prediction** — EC2, S3, Lambda, CloudWatch, etc.
- 📊 **IoT Core Cost Estimation** — Device usage, MQTT messages, policies, and rules.
- 🧠 **Hybrid Machine Learning Models** — gLSTM + Attention + XGBoost for AWS, LSTM for IoT.
- 💡 **Optimization Engine** — Rule-based cost-saving suggestions.
- ⚙️ **REST API** — Built using FastAPI.

---

---

## 🧠 ML Models

| Model                      | Use Case             | MAE   | RMSE  |
|---------------------------|----------------------|-------|-------|
| gLSTM + Attention         | AWS cost prediction  | 2.13  | 3.47  |
| XGBoost                   | AWS refinement       | 2.05  | 3.40  |
| LSTM                      | IoT cost prediction  | 1.85  | 2.92  |

---

## 🗃️ Data Format

### 📁 AWS Billing Data (`aws_billing_data.csv`)
| Column | Description |
|--------|-------------|
| region | e.g., `us-east-1` |
| ec2_instance_hours | Total EC2 usage hours |
| ... | ... |
| estimated_cost_usd | Target cost |

### 📁 IoT Usage Data (`iot_costs.csv`)
| Column | Description |
|--------|-------------|
| timestamp | Record time |
| thing_name | IoT device name |
| ... | ... |
| estimated_cost_usd | Target cost |

---

## 📦 Requirements

Install dependencies:
----
pip install -r requirements.txt
----
```bash

fastapi==0.95.2
uvicorn==0.22.0
python-dotenv==1.0.0
pydantic==1.10.7
numpy==1.23.5
pandas==2.0.1
scikit-learn==1.2.2
tensorflow==2.12.0
xgboost==1.7.5
matplotlib==3.7.1
seaborn==0.12.2
boto3==1.26.130
python-multipart==0.0.6
pytest==7.3.1
httpx==0.24.1
```
---
## 🔧 API Endpoints

### 📌 AWS Cost
- `POST /aws/predict` – Predict AWS service cost  
- `GET /aws/historical` – Retrieve historical AWS cost data  

### 📌 IoT Cost
- `POST /iot/predict` – Predict AWS IoT usage cost  
- `GET /iot/rule-costs` – List AWS IoT rule pricing  
- `GET /iot/policy-costs` – List AWS IoT policy pricing  

🧪 **Access Live API Docs**:  
[http://localhost:8000/docs](http://localhost:8000/docs) (via Swagger UI)

---
## Screenshots

Here is a preview of the app:

![Screenshot 2025-05-06 111950](https://github.com/user-attachments/assets/dc553a2d-1976-4dd1-828a-b02831429570)
![Screenshot 2025-05-07 115628](https://github.com/user-attachments/assets/8d8e96ca-e97a-4870-a58b-b2aead1a0cdd)


---
## 🧪 Training the Models

To train the ML models locally, run:

```bash
# Train all ML models
python -m src.models.model_training
```
---
## 🚀 Deployment

### ▶️ Local Development (FastAPI)

Run the FastAPI app locally with:

```bash
uvicorn src.main:app --reload
```
## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mukunthan**  
Email: mukunth.s1004@gmail.com  
GitHub: [Mukunth1004]((https://github.com/Mukunth1004))  



