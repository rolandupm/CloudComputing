import os
import sys
import keras
#import magic
import logging
import numpy as np
import urllib.request
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import MaxPooling2D
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import set_session
from flask import Flask, flash, request, redirect, render_template


sess = tf.InteractiveSession()

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.load_weights('model.h5')

#classifier = keras.models.load_model('model.h5')


graph = tf.get_default_graph()
logging.basicConfig(level=logging.DEBUG)
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = "super secret key"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')


@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			test_image = image.load_img(file, target_size = (64, 64))
			test_image = image.img_to_array(test_image)
			test_image = np.expand_dims(test_image, axis = 0)
			global graph,sess
			with sess.as_default():
				with graph.as_default():
					result = classifier.predict(test_image)
			print(file)
			if result[0][0] == 1:
				flash('dog')
			else:
				flash('cat')
			return redirect('/')
		else:
			flash('Allowed file types are png, jpg, jpeg')
			return redirect(request.url)


if __name__ == "__main__":
	app.config['SESSION_TYPE'] = 'filesystem'

	sess.init_app(app)
	app.debug = True
	app.run(debug=true)

