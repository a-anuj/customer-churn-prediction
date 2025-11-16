from flask import Flask,request,jsonify
import joblib
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = joblib.load("model.pkl")
preprocessing_pipeline = joblib.load("pipeline.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Apply preprocessing
        X = preprocessing_pipeline.transform(df)

        # Predict probability
        prob = model.predict_proba(X)[0][1]

        # Convert numpy -> python float
        prob_float = float(prob)

        # ----- AI Advice (Groq) -----
        prompt = f"""
        A telecom customer has a churn probability of {prob_float:.2f}.
        Give 3 clear retention actions.
        """
        ai_response = groq.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        advice_text = ai_response.choices[0].message['content']

        return jsonify({
            "churn_probability": prob_float,
            "advice": advice_text
        })

    except Exception as e:
        print("ERROR in /predict:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Churn Prediction model is running..."

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)