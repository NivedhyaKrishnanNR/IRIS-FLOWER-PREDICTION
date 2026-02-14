from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("iris_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    arr = np.array(values).reshape(1,-1)

    prediction = model.predict(arr)[0]

    flowers = ["Iris Setosa","Iris Versicolor","Iris Virginica"]
    result = flowers[prediction]

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)