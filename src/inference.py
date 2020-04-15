from tensorflow.keras.layers import Input, concatenate, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from custom_generator import DataGenerator
import matplotlib.pyplot as plt
import os
import csv
import PIL
import numpy as np
import random
import imutils
import cv2
import tensorflow as tf

TRAIN_PATH = '../datasets/data/nyu2_train.csv'
TEST_PATH = '../datasets/data/nyu2_test.csv'
MAX_DEPTH = 1000
MIN_DEPTH = 0
MAX_CLIP = 1
MIN_CLIP = 0
import tensorflow as tf


def read_csv(csv_file_path):
  with open(csv_file_path, 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    return [('../datasets/' + row[0], '../datasets/' + row[1]) for row in csv_reader if len(row) > 0]


if __name__ == "__main__":
  img_dm_pairs = read_csv('../datasets/data/nyu2_test.csv')
  labels = {i: j for i,j in img_dm_pairs}
  test_paths = [i for i,j in img_dm_pairs]
  partition = {'test': test_paths}

  arr = np.empty((32, 128, 128, 3))

  for i, ID in enumerate(partition['test'][:32]):
    arr[i,] = preprocess_image(ID, horizontal_flip=False)

  
  loaded_model = load_model('../models/model5.h5')
  # test_generator = DataGenerator(list_IDs=partition['test'], labels=labels, batch_size=32, dim=(128,128), n_channels=3, shuffle=False, pred=True)
  preds = loaded_model.predict(arr)

  for i in range(0, 15):
    path = partition['test'][i]
    label_path = labels[path]

    pred = preds[i]
    pred = np.squeeze(pred, axis=-1)
    print(pred.max(), pred.min())
    plt.subplot(1,3,1)
    plt.imshow(pred, cmap=plt.get_cmap('plasma_r'))

    plt.subplot(1,3,2)
    img = preprocess_depth_map(label_path, horizontal_flip=False)
    img = np.squeeze(img, axis=-1)
    plt.imshow(img, cmap=plt.get_cmap('plasma'))

    
    plt.subplot(1,3,3)
    img1 = preprocess_image(path, horizontal_flip=False)
    plt.imshow(img1)

    plt.show()