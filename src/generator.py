import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array, random_brightness, random_channel_shift
import cv2
import numpy as np
from rotation_utils import rotate_image

class DataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, list_IDs, rotate=True, preprocess_function=None, shuffle=True,
              show_intermediate=False, batch_size=32, dim=(224,224), regress=False):
    'Initialization'
    self.batch_size = batch_size
    self.dim = dim
    self.rotate = rotate
    self.preprocess_function = preprocess_function
    self.list_IDs = list_IDs
    self.shuffle = shuffle
    self.regress = regress
    self.show_intermediate = show_intermediate
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, 3))
    y = np.empty((self.batch_size, ))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      # Store sample
      
      img = load_img(ID)
      img = random_brightness(img, brightness_range=(0.75, 1.25))
      img = random_channel_shift(img, intensity_range=(0.5), channel_axis=2)
      img_array = img_to_array(img, dtype='float32')

      if self.rotate:
        angle = np.random.randint(360)
      else:
        angle = 0

      img_rotated = rotate_image(
        img_array,
        angle=angle,
        show=self.show_intermediate
      )

      img_processed = cv2.resize(img_rotated, self.dim)

      img_processed = img_processed / 255.

      img_new = array_to_img(img_processed)
      
      if self.preprocess_function is not None:
        img_new = self.preprocess_function(img_new)
      
            
      X[i,] = np.expand_dims(img_new, 0)
      y[i,] = angle
 
    # handle target variable (regression/classification)
    if not self.regress:
      y = to_categorical(y, 360)
    else:
      y = y / 360
    
    return X, y


if __name__ == "__main__":
  import glob
  import matplotlib.pyplot as plt
  images = glob.glob("images/*")
  print("Total Images:", len(images))
  gen = DataGenerator(images, rotate=True, batch_size=1, show_intermediate=True, preprocess_function=preprocess_input)
  batch_x, batch_y = gen[0]
  print(f"X.shape: {batch_x.shape} ; y.shape: {batch_y.shape}")
  plt.title(f"{np.argmax(batch_y)} degrees rotation")
  plt.imshow((batch_x[0]).astype('uint8'))
  plt.show()