#Importing Libraries
import re
import numpy as np
import os
from os.path import join, dirname, realpath
from flask import Flask, app, request, render_template
from keras import models
from keras.models import load_model
from keras.utils import load_img,img_to_array
from tensorflow.python.ops.gen_array_ops import concat
from keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename


#Loading the Model
model1 = load_model('level.h5')
model2 = load_model('body.h5')

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'uploads/') # where uploaded files are stored
ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF']) # models support png and gif as well

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # max upload - 10MB
app.secret_key = 'secret'

# check if an extension is valid and that uploads the file and redirects the user to the URL for the uploaded file
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/<a>')
def available(a):
	flash('{} coming soon!'.format(a))
	return render_template('index.html', result=None, scroll='third')

@app.route('/assessment')
def assess():
    return render_template('index.html', result=None, scroll='third')

@app.route('/assessment', methods=['GET', 'POST'])
def upload_and_classify():
    if request.method == 'POST':

        file = request.files['upload_image']
        filename = secure_filename(file.filename)  # used to secure a filename before storing it directly on the filesystem
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filepath)


        img1 = load_img(filepath, target_size=(224, 224))
        x = img_to_array(img1)
        x = np.expand_dims(x, axis=0)

        img_data1 = preprocess_input(x)

        img2 = load_img(filepath, target_size=(224, 224))
        y = img_to_array(img2)
        y = np.expand_dims(y, axis=0)
        img_data2 = preprocess_input(y)

        prediction1 = np.argmax(model1.predict(img_data1))
        prediction2 = np.argmax(model2.predict(img_data2))

        index1 = ['front', 'rear', 'side']
        index2 = ['minor', 'moderate', 'severe']

        result1 = index1[prediction1]
        result2 = index2[prediction2]
        if (result1 == "front" and result2 == "minor"):
            value = "3000 - 5000 INR"
        elif (result1 == "front" and result2 == "moderate"):
            value = "6000 - 8000 INR"
        elif (result1 == "front" and result2 == "severe"):
            value = "9000 - 11000 INR"
        elif (result1 == "rear" and result2 == "minor"):
            value = "4000 - 6000 INR"
        elif (result1 == "rear" and result2 == "moderate"):
            value = "7000 - 9000 INR"
        elif (result1 == "rear" and result2 == "severe"):
            value = "11000 - 13000 INR"
        elif (result1 == "side" and result2 == "minor"):
            value = "6000 - 8000 INR"
        elif (result1 == "side" and result2 == "moderate"):
            value = "9000 - 11000 INR"
        elif (result1 == "side" and result2 == "severe"):
            value = "12000 - 15000 INR"
        else:
            value = "16000 - 50000 INR"


        return render_template('results.html',result=value, location=result1,severity=result2, scroll='third', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Now one last thing is missing: the serving of the uploaded files.
# In the upload_file() we redirect the user to url_for('uploaded_file', filename=filename),
# that is, /uploads/filename. So we write the uploaded_file() function to return the file of that name.

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
