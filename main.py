from flask import Flask, url_for, request, render_template

#Py packages
import pandas as pd
import numpy as np

#ML packages

App = Flask(__name__)

@app.route('/')
def index():
  return render_template("index.html")

@app.route("/")
def predict():
  return render_template("predict.html")

If __name__ == '__main__':
  app.run(host='127.0.0.1',port=8080, debug=True)
