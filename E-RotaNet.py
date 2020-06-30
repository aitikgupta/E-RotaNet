import sys
sys.path.append('src')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from loss import angle_loss
from app import app
from predict import predict_single
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import random
from PIL import Image

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
		#print('upload_image filename: ' + filename)
		outputs = predict(save_path)
		# flash('Rotation by angle '+str(angle))
		return render_template('upload.html', filename=outputs)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


def predict(filepath):
	# print(filepath)
	rotation = random.choice([0, 45, 90, 135, 180, 225, 270, 315])

	args = {
		'image_path' : filepath,
		'model_path' : model,
		'rotation' : rotation,
		'regress' : False,
		'device' : 'cpu'
	}
	
	orig_img, output_img, angle = predict_single(args, show=False, crop=True)
	par_dir = os.path.abspath(os.path.join(filepath, os.pardir))
	filename = os.path.split(filepath)[1]
	rand_save_name = f"random_rotated_{rotation}_" + str(filename)
	output_save_name = f"output_rotated_{angle}_" + str(filename)

	orig_img.save(os.path.join(par_dir, rand_save_name))
	output_img.save(os.path.join(par_dir, output_save_name))
	
	intermediate_op = [rand_save_name, rotation]
	final_op = [output_save_name, angle]

	return [filename, intermediate_op, final_op]

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()