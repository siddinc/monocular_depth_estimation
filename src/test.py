from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy, binary_crossentropy
import numpy as np
import tensorflow as tf
import cv2

# binary_crossentropy = BinaryCrossentropy()


def DiceBCELoss(y_true, y_pred, smooth=1e-7):
  inputs = y_true
  targets = y_pred

  BCE = binary_crossentropy(targets, inputs)
  intersection = K.sum(K.dot(targets, inputs))
  total = K.sum(targets) + K.sum(inputs)
  dice_loss = 1.0 - ((2*intersection + smooth) / (total + smooth))

  Dice_BCE = BCE + dice_loss
  return Dice_BCE


def depth_loss(y_true, y_pred):
  w1, w2, w3 = 1.0, 1.0, 0.1
  w4, w5 = 1.0, 1.0

  l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

  l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

  return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


if __name__ == "__main__":
  pass
