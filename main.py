from flask import Flask, url_for, request, render_template

#Py packages
import pandas as pd
import numpy as np
import os

#ML packages
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



App = Flask(__name__)

@app.route('/')
def index():
  return render_template("index.html")

@app.route("/", methods = ['POST'])
def predict():
  url="https://raw.githubusercontent.com/rsikder007/myrepo/master/train.csv"
  data=pd.read_csv(url)
  data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace=True)
  data['Self_Employed'].fillna('No',inplace=True)
  data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
  data['Married'].fillna(data['Married'].mode()[0], inplace=True)
  data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
  data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
  data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
  
  encod = LabelEncoder()
  data_var = ['Education','Self_Employed','Property_Area','Gender','Dependents','Married','Loan_Status']
  
  for variable in data_var:
    data[variable] = encod.fit_transform(data[variable])
  
  x_var = data[['Education','Gender','Credit_History','Married']]
  y_var = data['Loan_Status']
  
  x_train,x_test, y_train, y_test = train_test_split(x_var,y_var,train_size=0.80, random_state = 30)
  
  loan_model = DecisionTreeClassifier()
  loan_model.fit(x_train,y_train)
  #loan_prediction = loan_model.predict(x_test)
  accuracy_score(loan_prediction, y_test)
  
  if request.method == 'POST':
    comment = request.form['comment']
    data = [comment]
    vect = encod.fit_transform(data).toarray()
    loan_pred = loan_model.predict(vect)
  return render_template("predict.html", prediction = loan_pred, comment = comment)

If __name__ == '__main__':
  app.run(host='127.0.0.1',port=8080, debug=True)
