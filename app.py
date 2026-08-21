import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# Load the trained Keras model
# Since model.pkl contains the saved Keras Sequential architecture and weights
MODEL_PATH = "model.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Churn Prediction ANN API is running."
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded properly."}), 500

    try:
        data = request.get_json(force=True)

        # Your input layer expects 10 feature inputs
        features = np.array(data["features"], dtype=np.float32)

        # Reshape for single sample prediction if 1D array is provided
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)

        # Run model inference
        prediction = model.predict(features)
        churn_probability = float(prediction[0][0])
        will_churn = bool(churn_probability > 0.5)

        return jsonify({
            "churn_probability": churn_probability,
            "will_churn": will_churn
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
