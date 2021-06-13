from flask import Flask, flash, render_template, url_for, request, jsonify, redirect
import pandas as pd
from werkzeug.utils import secure_filename
import cv2 as cv
import keras
import numpy as np
from keras.models import load_model
import os
from keras import backend as K

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
			result = predict_image(image)
			redirect(url_for('upload_file',filename=filename, result = result))
			return render_template('result.html', result = result)
	return render_template('home.html')

def predict_image(image):
	classifier = load_model('./model/OCT-DenseNetModel.h5')
	image = cv.resize(image, (224,224), 3)
	image = image / 255
	image = np.expand_dims(image, axis=0)
	res = classifier.predict(image)

	classes_list = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
	predicted_class_index = np.argmax(res,axis=1)
	predicted_class = classes_list[predicted_class_index[0]]
	K.clear_session()
	return predicted_class


if __name__ == '__main__' :
    app.run(debug=True)
