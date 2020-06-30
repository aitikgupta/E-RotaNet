import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import glob

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tqdm import tqdm
import numpy as np

from generator import DataGenerator
from loss import angle_loss
from preprocess import preprocess_single_img

def evaluate_dir_random_rotation(args):
    '''
    Evaluates an image directory, randomly rotating images on-the-fly
    '''
    images = glob.glob(os.path.join(args['image_dir'], "*.jpg"))

    # Creating test generator
    test_gen = DataGenerator(
        images,
        rotate=True,
        preprocess_function=preprocess_input,
        shuffle=True,
        show_intermediate=False,
        batch_size=args['batch_size'],
        dim=args['img_size'],
        regress=args['regress']
    )

    # Loading model
    model = load_model(args['model_dir'], custom_objects={"angle_loss": angle_loss})

    # Running evaluation
    out = model.evaluate(
        test_gen,
        steps = int(len(images) / args['batch_size'])
    )

    print(f"Test Loss: {out[0]} ; Angle Loss: {out[1]}")

def evaluate_single_with_all_rotations(args):
    '''
    Evaluates a single image, from 0->360 degrees rotation, and prints the mean angle_loss
    '''
    # Loading the model
    model = load_model(args['model_dir'], custom_objects={"angle_loss": angle_loss})
    
    total_angle_loss = 0
    
    # Each iteration calculates error for i-th angle 
    for current_rotation_angle in tqdm(range(360), total=360):

        img_processed = preprocess_single_img(args['eval_single'],
                            rotation=current_rotation_angle,   # Rotating by current_rotation_angle                            show=False,
                            crop=True,
                            dim=args['img_size'],
                            preprocess_function=preprocess_input
                            )[0]

        inp = np.expand_dims(img_processed, 0)
        pred = model.predict(inp)
        pred_angle = np.argmax(pred)

        total_angle_loss += abs(pred_angle - current_rotation_angle)
    
    print("Total Angle Loss:", total_angle_loss/360.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Path to model', default='../release/model.h5')
    parser.add_argument('--eval_dir', help='Path to images directory (Images will be randomly rotated)', default=None)
    parser.add_argument('--eval_single', help='Path to the image (Image will be rotated from 0 to 360 degrees)', default=None)
    parser.add_argument('--batch_size', help='Batch size', default=32)
    parser.add_argument('--img_size', help='Input size of image to the model', default=(224,224))
    parser.add_argument('--regress', help='Use regression instead of classification', action='store_true')
    parser.add_argument('--device', help="Use device for inference (gpu/cpu)", default='cpu')

    args = parser.parse_args()

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        assert args.eval_dir is not None and args.eval_single is not None
    except AssertionError as e:
        e.args = ["Both --eval_dir and --eval_single can't be None, Run 'python evaluate.py --help' for more information"]
        raise

    if args.eval_single is not None:
        evaluate_single_with_all_rotations(args)
    else:
        evaluate_dir_random_rotation(args)