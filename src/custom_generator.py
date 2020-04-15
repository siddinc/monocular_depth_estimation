from tensorflow.keras.utils import Sequence
import numpy as np
import imutils
import cv2
import random
from skimage.transform import resize


def resize_img(img, height=128):
  resized_img = resize(img, (height, int(height*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True)
  return resized_img

def preprocess_image(img_path, horizontal_flip=None):
  image = cv2.imread(img_path, cv2.IMREAD_COLOR)
  if horizontal_flip:
    image = cv2.flip(image, 1)
  image = resize_img(image, height=128)
  image = np.clip(image.astype(np.float64)/255, 0, 1)
  image = image[:, 21:149, :]
  return image

def preprocess_depth_map(depth_map_path, horizontal_flip):
  depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
  if horizontal_flip:
    depth_map = cv2.flip(depth_map, 1)
  depth_map = resize_img(depth_map, height=128)
  depth_map = 1000/np.clip(depth_map.astype(np.float64)/255*1000, 0, 1000)
  depth_map = depth_map[:, 21:149]
  depth_map = np.reshape(depth_map, (128,128,1))
  return depth_map


class DataGenerator(Sequence):
  def __init__(self, list_IDs, labels, batch_size=8, dim=(128,128), n_channels=3, shuffle=True, pred=False):
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.shuffle = shuffle
    self.pred = pred
    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    if self.pred:
      X = self.__data_generation(list_IDs_temp)
      return X
    X, y = self.__data_generation(list_IDs_temp)
    return X, y

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    X = np.empty((self.batch_size, 128, 128, 3))

    if not self.pred:
      y = np.empty((self.batch_size, 128, 128, 1))

      for i, ID in enumerate(list_IDs_temp):
        res = random.choice([True, False])
        X[i,] = preprocess_image(ID, res)
        y[i,] = preprocess_depth_map(self.labels[ID], res)
      return X, y
    else:
      for i, ID in enumerate(list_IDs_temp):
        res = random.choice([True, False])
        X[i,] = preprocess_image(ID, res)
      return X