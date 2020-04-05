from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import PIL
import cv2
import json
import csv
import tensorflow as tf
import progressbar
import numpy as np
from HDF5_dataset_writer import HDF5DatasetWriter

TRAIN_PATH = os.path.abspath('../datasets/data/nyu2_train.csv')
TEST_PATH = os.path.abspath('../datasets/data/nyu2_test.csv')
TRAIN_HDF5 = os.path.abspath('../datasets/data/hdf5/train.hdf5')
VAL_HDF5 = os.path.abspath('../datasets/data/hdf5/val.hdf5')
TEST_HDF5 = os.path.abspath('../datasets/data/hdf5/test.hdf5')
DATASET_MEAN = os.path.abspath('../datasets/data/mean.json')
IMG_SHAPE = (480, 640, 3)
DEPTH_IMG_SHAPE = (240, 320, 1)


def read_csv(csv_file_path):
  with open(csv_file_path, 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    return [row for row in csv_reader if len(row) > 0]

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



def resize_img(img, height=480, padding=6):
  resized_img = resize(img, (height, int(height*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True)
  return resized_img


if __name__ == '__main__':
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_img_paths(TRAIN_PATH, TEST_PATH)
  print(x_train[:3])
  print(x_val[:3])
  print(x_test[:3])

  datasets = [
    ('train', x_train, y_train, TRAIN_HDF5),
    ('val', x_val, y_val, VAL_HDF5),
    ('test', x_test, y_test, TEST_HDF5),
  ]

  (R, G, B) = ([], [], [])

  for (dType, paths, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    # writer = HDF5DatasetWriter(image_dims=(len(paths), 480, 640, 3), depth_map_dims=(len(paths), 240, 320, 1), outputPath=outputPath)
    # widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
    #            progressbar.Bar(), " ", progressbar.ETA()]
    # pbar = progressbar.ProgressBar(maxval=len(paths),
    #                                widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        img = cv2.imread(path)
        print(type(img))
        # depth_map = cv2.imread(label)
        # print(depth_map.shape)
    #     rgb = tf.image.convert_image_dtype(img, dtype=tf.float32)

    #     depth = tf.image.convert_image_dtype(depth_map / 255.0, dtype=tf.float32)
    #     depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)


    #     if dType == "train":
    #         (b, g, r) = cv2.mean(img)[:3]
    #         R.append(r)
    #         G.append(g)
    #         B.append(b)

    #     writer.add([img], [depth])
    #     pbar.update(i)
    # pbar.finish()
    # writer.close()

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}

with open(DATASET_MEAN, "w") as f:
  f.write(json.dumps(D))