from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.applications.efficientnet import preprocess_input
from rotation_utils import rotate_image
import cv2
import numpy as np

def preprocess_single_img(filepath, rotation=0, show=False, crop=True, dim=(224,224), preprocess_function=None):
    '''
    Preprocesses a single image, given the full path to the image
    '''
    img = load_img(filepath)
    img_array = img_to_array(img, dtype='float32')

    orig_img_size = img_array.shape

    rotated = rotate_image(
        img_array,
        angle=rotation,
        show=show,
        crop=crop
    )

    img_res = cv2.resize(rotated, dim)
    img_res = img_res / 255.

    img_processed = array_to_img(img_res)
    
    if preprocess_function is not None:
        img_processed = preprocess_input(img_processed)

    return [img_processed, orig_img_size]


if __name__ == "__main__":
    _ = preprocess_single_img("images/000001_0.jpg",
                            rotation=45,
                            show=True,
                            crop=False,
                            dim=(224,224),
                            preprocess_function=preprocess_input)
