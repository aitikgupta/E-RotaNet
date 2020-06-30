import sys
sys.path.append('src')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random

import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template
from tensorflow.keras.models import load_model

from predict import predict_single
from loss import angle_loss
from app import app

# Loading the model
model = load_model("release/model.h5", custom_objects={'angle_loss': angle_loss})

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(save_path)
		outputs = predict(save_path)
		flash('Images saved at: ' + app.config['UPLOAD_FOLDER'])
		return render_template('upload.html', filename=outputs)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

def predict(filepath):
	# Random rotation of images
	rotation = random.choice([0, 45, 90, 135, 180, 225, 270, 315])

	args = {
		'image_path' : filepath,
		'model_path' : model,
		'rotation' : rotation,
		'regress' : False,
		'device' : 'cpu'	# Letting deployed app use cpu
	}
	
	# Getting output from the model
	orig_img, output_img, angle = predict_single(args, show=False, crop=True)
	
	# Saving the randomly rotated images, along with output images.
	par_dir = os.path.abspath(os.path.join(filepath, os.pardir))
	filename = os.path.split(filepath)[1]
	rand_save_name = f"random_rotated_{rotation}_" + str(filename)
	output_save_name = f"output_rotated_{angle}_" + str(filename)

	orig_img.save(os.path.join(par_dir, rand_save_name))
	output_img.save(os.path.join(par_dir, output_save_name))
	
	# Intermediate and final outputs combined
	intermediate_op = [rand_save_name, rotation]
	final_op = [output_save_name, angle]

	return [filename, intermediate_op, final_op]

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()