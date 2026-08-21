from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load ANN model
with open("model(2).pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Get input values from HTML
            input1 = float(request.form["input1"])
            input2 = float(request.form["input2"])
            input3 = float(request.form["input3"])
            input4 = float(request.form["input4"])

            # Create input array
            data = np.array([[input1, input2, input3, input4]])

            # Prediction
            result = model.predict(data)

            # For binary classification
            prediction = "Positive" if result[0][0] >= 0.5 else "Negative"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
