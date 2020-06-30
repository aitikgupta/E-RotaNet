import os
import random
import argparse

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

from preprocess import preprocess_single_img
from rotation_utils import rotate_image


def predict_single(args, show=False, crop=True):
    '''
    Runs inference on a single image, returns original image, output image and predicted angle
    Note: filepath contains the full path of an image (not just parent directory)
    '''
    if args['rotation'] == -1:
        args['rotation'] = random.randint(0, 360)

    img_processed, orig_img_size = preprocess_single_img(args['image_path'], rotation=args['rotation'],
                                                    crop=crop, show=show, preprocess_function=preprocess_input)
    
    # input_dimension: (batch_size, height, width, channels)
    input_to_model = np.expand_dims(img_processed, 0)
    pred = args['model_path'].predict(input_to_model)

    if args['regress']:
        pred_angle = pred*360
    else:
        pred_angle = np.argmax(pred)
    
    img_processed = img_to_array(img_processed)
    
    # Rotate the image according to the predicted angle
    rotated_img = rotate_image(
        img_processed,
        angle=-pred_angle,
        show=show,
        crop=crop
    )

    # Resize output image to match input dimensions
    orig_img = cv2.resize(img_processed, orig_img_size[:2])
    output_img = cv2.resize(rotated_img, orig_img_size[:2])

    # Converting to PIL Image objects
    orig_img = Image.fromarray(orig_img.astype('uint8'))
    output_img = Image.fromarray(output_img.astype('uint8'))

    return [orig_img, output_img, pred_angle]

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    from loss import angle_loss

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='Path to the image', default='images/000001_0.jpg')
    parser.add_argument('--model_path', help='Path to the model', default='release/model.h5')
    parser.add_argument('--rotation', type=int, help='Rotation amount (-1 for random)', default=-1)
    parser.add_argument('--regress', help='Use regression instead of classification', action='store_true')
    parser.add_argument('--device', help="Use device for inference (gpu/cpu)", default='cpu')

    args = parser.parse_args()

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    model = load_model(args.model_path, custom_objects={'angle_loss': angle_loss})
    args.model_path = model

    _ = predict_single(args.__dict__, show=True, crop=True)
