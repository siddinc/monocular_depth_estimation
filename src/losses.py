import numpy as np
import cv2
from skimage.transform import resize
from tensorflow.keras import backend as K
import tensorflow as tf


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


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
  
  # Point-wise depth
  l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

  # Edges
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

  # Structural similarity (SSIM) index
  l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

  # Weights
  w1 = 1.0
  w2 = 1.0
  w3 = theta
  return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

def depth_acc(y_true, y_pred):
  return 1.0 - depth_loss_function(y_true, y_pred)