# 📊 Customer Churn Prediction + AI Retention Assistant  
A production-ready ML app that predicts telecom customer churn **and generates AI-powered retention advice** using **Groq LLM**.

This project includes:

- End-to-end ML pipeline (preprocessing + model)
- Churn probability prediction API (`/predict`)
- AI retention recommendations using Groq
- Flask backend (Dockerized)
- Railway deployment

## 🧠 Tech Stack
- Flask (Python)
- scikit-learn
- pandas 
- numpy
- Groq LLM API
- Docker
- Railway (Deploying)

## 🚀 Features

### 🔍 **1. Churn Prediction**
- Preprocessing with ColumnTransformer
- OneHotEncoder for categoricals
- StandardScaler for numeric columns
- Model trained using:
  - Logistic Regression (best performing)
  - Saved using joblib for production

### 🤖 **2. AI-Powered Retention Advice**
Powered by **Groq Llama 3.1 models**  
The API returns clear business actions to reduce churn.



## 📖 API Documentation

### Predict Churn

Predicts the churn probability for a single customer.

* **URL:** `/predict`
* **Method:** `POST`
* **Content-Type:** `application/json`

#### Request Body
The request must be a JSON object containing the customer's features.


**Example JSON Input:**
```json
{
"gender": "Female",
"SeniorCitizen": 0,
"Partner": "Yes",
"Dependents": "No",
"tenure": 12,
"PhoneService": "Yes",
"MultipleLines": "No",
"InternetService": "Fiber optic",
"OnlineSecurity": "No",
"OnlineBackup": "No",
"DeviceProtection": "Yes",
"TechSupport": "No",
"StreamingTV": "Yes",
"StreamingMovies": "Yes",
"Contract": "Month-to-month",
"PaperlessBilling": "Yes",
"PaymentMethod": "Electronic check",
"MonthlyCharges": 70.5,
"TotalCharges": 845.0
}
```
**Example HTTP Response:**
```json
{
"churn_probability": 0.73,
  "advice": "1. Offer a contract upgrade incentive...\n2. Reduce monthly charges...\n3. Provide proactive tech support..."
}
```