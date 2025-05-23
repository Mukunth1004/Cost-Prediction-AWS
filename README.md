
# 🌐 AWS IoT Cost Prediction
````markdown
A Machine learning-powered cost prediction system for AWS services and IoT Core devices. Features include real-time predictions, cost breakdowns, optimization recommendations, and an AWS-style dashboard interface.
````
---

## ✨ Key Features
````markdown
- 🔍 AWS Cost Prediction — Forecast costs for EC2, S3, Lambda, and more.
- 📡 IoT Core Cost Prediction — Estimate messaging, shadows, rules, and data transfer costs.
- 📊 Detailed Cost Breakdown — Transparent, component-wise billing breakdown.
- 💡 Optimization Suggestions — Actionable tips to reduce cloud spending.
- 🖥️ AWS-Style UI — Clean interface modeled after AWS Console.
- 🔗 RESTful API — Fully documented for easy integration.
````
---

## ⚙️ Technology Stack
````markdown
| Layer        | Technologies                               |
|--------------|--------------------------------------------|
| Frontend     | HTML5, CSS3, JavaScript, AWS UI Components |
| Backend      | Python, FastAPI                            |
| ML Models    | XGBoost (AWS), TensorFlow (IoT)            |
| Data         | Pandas, NumPy, Scikit-learn                |
````
---

## 📦 Setup Instructions

### ✅ Prerequisites
````markdown
- Python 3.8+
- `pip`
- `virtualenv` (recommended)
````
---

### 🧰 Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # For Linux/macOS
venv\Scripts\activate       # For Windows
```

---

### 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔐 Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and config
```

---

## ▶️ Running the App


### 🚀 Start the FastAPI Server

```bash
uvicorn src.main:app --reload
```

---

### 🌐 Visit the App in Browser

```
http://localhost:8000
http://localhost:8000/docs
```

---

## 🚀 API Endpoints
````markdown
| Endpoint           | Method | Description       |
|--------------------|--------|-------------------|
| `/api/predict/aws` | POST   | Predict AWS costs |
| `/api/predict/iot` | POST   | Predict IoT costs |
| `/api/train/aws`   | POST   | Retrain AWS model |
| `/api/train/iot`   | POST   | Retrain IoT model |
````
---

## 🧠 Machine Learning Models



### 🔸 AWS Cost Prediction
````markdown
- Model: XGBoost Regressor  
- Features:
  - EC2 hours
  - S3 storage
  - Lambda invocations
  - And more  
- Target: `estimated_cost_usd`
````
---

### 🔹 IoT Cost Prediction
````markdown
- Model: TensorFlow Neural Network  
- Features:
  - MQTT messages
  - Shadow updates
  - Rules count
  - And more  
- Target: `estimated_cost_usd`
````
---

## 🛠️ Contributing

We welcome contributions! To contribute:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add AmazingFeature"
   ```
4. **Push to GitHub**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

---

## 📜 License
````markdown
Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.
````
---

## 📸 Screenshots

![Screenshot 2025-05-23 152554](https://github.com/user-attachments/assets/e2655b9f-38b7-43e0-9448-f091f8f22358)


---

