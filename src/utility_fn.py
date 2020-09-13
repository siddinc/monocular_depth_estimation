from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import PIL
import cv2
import json
import csv
import progressbar
import numpy as np
import random

TRAIN_PATH = os.path.abspath('../datasets/data/nyu2_train.csv')
TEST_PATH = os.path.abspath('../datasets/data/nyu2_test.csv')
IMG_SHAPE = (480, 640, 3)
DEPTH_IMG_SHAPE = (240, 320, 1)
MAX_DEPTH = 1000
MIN_DEPTH = 0
MAX_CLIP = 1
MIN_CLIP = 0
import tensorflow as tf


def read_csv(csv_file_path):
  with open(csv_file_path, 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    return [(os.path.abspath('../datasets/' + row[0]), os.path.abspath('../datasets/' + row[1])) for row in csv_reader if len(row) > 0]


def load_img_paths(train_path, test_path):
  train_mapping = read_csv(train_path)
  test_mapping = read_csv(test_path)

  x_train, x_test, y_train, y_test = [], [], [], []

  for i,j in train_mapping:
    x_train.append(i)
    y_train.append(j)

  for i,j in test_mapping:
    x_test.append(i)
    y_test.append(j)

  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
  return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


def resize_img(img, height=480):
  resized_img = resize(img, (height, int(height*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True)
  return resized_img


def preprocess_image(img_path):
  image = load_img(img_path, color_mode='rgb')
  image = img_to_array(image, dtype='float')
  image = resize_img(image, height=480)
  image = tf.clip_by_value(image / 255, MIN_CLIP, MAX_CLIP)
  return image


def preprocess_depth_map(depth_map_path):
  depth_map = load_img(depth_map_path, color_mode='grayscale')
  depth_map = img_to_array(depth_map, dtype='float')
  depth_map = resize_img(depth_map, height=240)
  depth_map = tf.clip_by_value(depth_map/255*MAX_DEPTH, MIN_DEPTH, MAX_DEPTH)
  depth_map = MAX_DEPTH / depth_map
  return depth_map


if __name__ == '__main__':
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_img_paths(TRAIN_PATH, TEST_PATH)