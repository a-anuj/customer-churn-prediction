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
pipeline = joblib.load("pipeline.pkl")

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.get_json()
        df = pd.DataFrame([data])

        x = pipeline.transform(df)

        prob = model.predict_proba(x)[0][1]


        prompt = f"""
                You are an expert customer retention specialist.

                The following is a telecom customer profile:

                {data}

                Their predicted churn probability is: {churn_prob_float:.2f}

                Explain the top 3 reasons WHY they might churn.
                Then give 4 simple, actionable retention strategies the company should apply.

                Keep the answer very concise, direct, and practical.
                """
        
        response = groq.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        advice = response.choices[0].message.content


        return jsonify({
            "churn probability":float(prob),
            "retention advice":advice
        })
    except Exception as e:
        return jsonify({"error":Exception})

@app.route('/')
def home():
    return "Churn Prediction model is running..."

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)