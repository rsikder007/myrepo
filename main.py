import flask
from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf


App = Flask(__name__)

@app.route('/')
def hello():
	return flask.render_template('index.html')


def image_preprocessor(pics):
	img = np.asarray(pics)
	img = img / 255.0
	img = img[np.newaxis, ..., np.newaxis]
	img = tf.image.resize(img,[28,28])
	return img

def value_prediction(to_predict):
	loaded_model = keras.models.load_model("trained_model.h5")
	pred = loaded_model(to_predict)
	pred = tf.nn.softmax(pred)
	pred_0 = pred[0]
	label_0 = np.argmax(pred_0)
	return label_0

@app.route('/predict', methods = ['POST'])
def result():
	if request.method == 'POST':
		file = request.files.get('file')
		tensor_0 = file.read()
		tensor_1 = Image.open(io.BytesIO(tensor_0)).convert('L')
		tensor_2 = image_preprocessor(tensor_1)
		result = value_prediction(tensor_2)
		result_2 = {"prediction":int(result)}
		return jsonify(result_2)
		

If __name__ == '__main__':
	app.run(host='127.0.0.1',port=8080, debug=True	
