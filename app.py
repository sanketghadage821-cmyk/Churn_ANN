from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ANN model
with open("model(2).pkl", "rb") as file:
    model = pickle.load(file)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .card {
            width: 450px;
            background: white;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }

        h1 {
            text-align: center;
            color: #203a43;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 25px;
        }

        label {
            display: block;
            margin-top: 14px;
            margin-bottom: 6px;
            font-weight: bold;
            color: #333;
        }

        input {
            width: 100%;
            padding: 11px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 15px;
        }

        input:focus {
            outline: none;
            border: 2px solid #2c5364;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 13px;
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
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            background: #e8f5e9;
            color: #1b5e20;
            font-weight: bold;
        }

        .error {
            margin-top: 20px;
            padding: 12px;
            background: #ffebee;
            color: #c62828;
            border-radius: 8px;
            text-align: center;
        }
    </style>
</head>

<body>

<div class="card">

    <h1>Customer Churn Prediction</h1>

    <p class="subtitle">
        Artificial Neural Network
    </p>

    <form method="POST">

        <label>Input 1</label>
        <input type="number" step="any" name="input1" required>

        <label>Input 2</label>
        <input type="number" step="any" name="input2" required>

        <label>Input 3</label>
        <input type="number" step="any" name="input3" required>

        <label>Input 4</label>
        <input type="number" step="any" name="input4" required>

        <button type="submit">Predict Churn</button>

    </form>

    {% if prediction %}
        <div class="result">
            {{ prediction }}
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


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            input1 = float(request.form["input1"])
            input2 = float(request.form["input2"])
            input3 = float(request.form["input3"])
            input4 = float(request.form["input4"])

            # Input data
            data = np.array([
                [input1, input2, input3, input4]
            ])

            # ANN prediction
            result = model.predict(data, verbose=0)

            # Binary classification
            probability = float(result[0][0])

            if probability >= 0.5:
                prediction = "⚠️ Customer is likely to CHURN"
            else:
                prediction = "✅ Customer is likely to STAY"

        except Exception as e:
            error = "Prediction Error: " + str(e)

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
