from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 1️⃣ Load the trained model
import os
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "energy_model.pkl")
model = joblib.load(os.path.abspath(model_path))
print("✅ Model loaded successfully!")

@app.route("/")
def home():
    return jsonify({"message": "⚡ Smart Energy Predictor API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # 2️⃣ Convert input JSON into DataFrame
        features = pd.DataFrame([data])

        # 3️⃣ Predict using the trained model
        prediction = model.predict(features)[0]

        return jsonify({
            "predicted_electricity_demand": round(float(prediction), 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
