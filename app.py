from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model and scaler (for "Amount")
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")  # This scaler was trained only on "Amount"

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]  # Get input features
        data = np.array(data)  # Convert to numpy array

        amount_index = -1  # Assuming "Amount" is the last feature in input

        # Scale only the "Amount" feature
        data[amount_index] = scaler.transform([[data[amount_index]]])[0][0]

        # Reshape for prediction
        data_scaled = data.reshape(1, -1)

        # Make prediction
        prediction = model.predict(data_scaled)[0]

        return jsonify({"fraud": bool(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
