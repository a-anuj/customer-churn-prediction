from flask import Flask,request,jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.get_json()
        df = pd.DataFrame([data])

        x = pipeline.transform(df)

        prob = model.predict_proba(x)[0][1]

        return jsonify({
            "churn probability":float(prob)
        })
    except Exception as e:
        return jsonify({"error":Exception})

@app.route('/')
def home():
    return "Churn Prediction model is running..."

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)