from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# -------------------------------------------------
# Load model
# -------------------------------------------------

MODEL_PATH = "model(2).pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Model loaded successfully")

except Exception as e:
    model = None
    print("Model loading error:", e)


# -------------------------------------------------
# HTML
# -------------------------------------------------

HTML = """
<!DOCTYPE html>
<html>
<head>

    <title>Customer Churn Prediction</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            min-height: 100vh;

            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 90%;
            max-width: 500px;
            background: white;
            padding: 35px;
            border-radius: 20px;

            box-shadow:
                0 15px 35px rgba(0,0,0,0.3);
        }

        h1 {
            text-align: center;
            color: #203a43;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 30px;
        }

        .input-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            font-weight: bold;
            color: #333;
            margin-bottom: 6px;
        }

        input {
            width: 100%;
            padding: 12px;

            border: 1px solid #ccc;
            border-radius: 8px;

            font-size: 15px;
        }

        input:focus {
            outline: none;
            border: 2px solid #203a43;
        }

        button {
            width: 100%;
            padding: 14px;

            margin-top: 10px;

            border: none;
            border-radius: 8px;

            background: #203a43;
            color: white;

            font-size: 17px;
            font-weight: bold;

            cursor: pointer;
        }

        button:hover {
            background: #0f2027;
        }

        .result {
            margin-top: 25px;
            padding: 18px;

            border-radius: 10px;

            background: #e8f5e9;
            color: #1b5e20;

            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }

        .error {
            margin-top: 20px;
            padding: 15px;

            border-radius: 10px;

            background: #ffebee;
            color: #c62828;

            text-align: center;
        }

        .probability {
            margin-top: 8px;
            font-size: 14px;
            font-weight: normal;
        }

    </style>

</head>

<body>

<div class="container">

    <h1>Customer Churn Prediction</h1>

    <p class="subtitle">
        Artificial Neural Network
    </p>

    <form method="POST">

        <div class="input-group">
            <label>Input 1</label>
            <input
                type="number"
                step="any"
                name="input1"
                required>
        </div>

        <div class="input-group">
            <label>Input 2</label>
            <input
                type="number"
                step="any"
                name="input2"
                required>
        </div>

        <div class="input-group">
            <label>Input 3</label>
            <input
                type="number"
                step="any"
                name="input3"
                required>
        </div>

        <div class="input-group">
            <label>Input 4</label>
            <input
                type="number"
                step="any"
                name="input4"
                required>
        </div>

        <button type="submit">
            Predict Churn
        </button>

    </form>


    {% if prediction %}

        <div class="result">

            {{ prediction }}

            {% if probability %}
                <div class="probability">
                    Probability: {{ probability }}
                </div>
            {% endif %}

        </div>

    {% endif %}


    {% if error %}

        <div class="error">
            {{ error }}
        </div>

    {% endif %}

</div>

</body>
</html>
"""


# -------------------------------------------------
# Home / Prediction
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability_text = None
    error = None

    if request.method == "POST":

        try:

            if model is None:
                raise Exception("Model could not be loaded.")

            # Get input values
            input1 = float(request.form["input1"])
            input2 = float(request.form["input2"])
            input3 = float(request.form["input3"])
            input4 = float(request.form["input4"])

            # Create NumPy array
            input_data = np.array(
                [[
                    input1,
                    input2,
                    input3,
                    input4
                ]],
                dtype=np.float32
            )

            # Make prediction
            result = model.predict(input_data, verbose=0)

            print("Prediction result:", result)

            # Binary classification
            probability = float(np.asarray(result).flatten()[0])

            probability_text = f"{probability:.2%}"

            if probability >= 0.5:

                prediction = "⚠️ Customer is likely to CHURN"

            else:

                prediction = "✅ Customer is likely to STAY"


        except Exception as e:

            print("Prediction error:", e)

            error = str(e)


    return render_template_string(
        HTML,
        prediction=prediction,
        probability=probability_text,
        error=error
    )


# -------------------------------------------------
# Health check
# -------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "OK",
        "model_loaded": model is not None
    }


# -------------------------------------------------
# Render
# -------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
