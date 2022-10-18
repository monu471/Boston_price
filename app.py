import pickle
import re
from django.shortcuts import render
from flask import Flask,request,jsonify,app,url_for,render_template
import numpy as np
import pandas as pd
app = Flask(__name__)
reg_model = pickle.load(open("boston_prediction.pkl","rb"))
scaler = pickle.load(open("scaling.pkl","rb"))
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict_api",methods = ["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = reg_model.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route("/predict",methods = ["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = reg_model.predict(final_input)[0]
    return render_template("home.html",prediction_text = " the predicted house price is :-{}".format(output))



if __name__ == "__main__":
    app.run(port=5003,debug=True)
    