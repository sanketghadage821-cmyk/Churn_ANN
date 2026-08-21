import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Path to your saved Keras model file
MODEL_PATH = "model.pkl"

# Load model at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Churn Prediction API is up and running."
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on server."}), 500

    try:
        data = request.get_json(force=True)

        # Ensure features array is present in request body
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request JSON."}), 400

        # Input array conversion (expects 10 numerical features)
        features = np.array(data["features"], dtype=np.float32)

        # Convert 1D input array [f1, f2, ...] into 2D batch [[f1, f2, ...]]
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)

        # Perform inference
        prediction = model.predict(features)
        churn_probability = float(prediction[0][0])
        will_churn = bool(churn_probability > 0.5)

        return jsonify({
            "churn_probability": round(churn_probability, 4),
            "will_churn": will_churn
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
