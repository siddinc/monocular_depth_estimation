from tensorflow.keras.layers import Input, concatenate, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import tensorflow as tf

def downsampling_block(input_tensor, n_filters):
  x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(input_tensor)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)
  return x

def upsampling_block(input_tensor, n_filters, name, concat_with):
  x = UpSampling2D((2, 2), interpolation='bilinear', name=name)(input_tensor)
  x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name+"_convA")(x)
  x = LeakyReLU(alpha=0.2)(x)

  x = concatenate([x, concat_with], axis=3)

  x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name+"_convB")(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)

  x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name+"_convC")(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = BatchNormalization()(x)
  return x

def build(height, width, depth):
  # input
  i = Input(shape=(height, width, depth))

  # encoder
  conv1 = downsampling_block(i, 32)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = downsampling_block(pool1, 64)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = downsampling_block(pool2, 128)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = downsampling_block(pool3, 256)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  # bottleneck
  conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
  conv5 = LeakyReLU(alpha=0.2)(conv5)
  conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
  conv5 = LeakyReLU(alpha=0.2)(conv5)

  # decoder
  conv6 = upsampling_block(conv5, 256, "up1", concat_with=conv4)
  conv7 = upsampling_block(conv6, 128, "up2", concat_with=conv3)
  conv8 = upsampling_block(conv7, 64, "up3", concat_with=conv2)
  conv9 = upsampling_block(conv8, 32, "up4", concat_with=conv1)

  # output
  o = Conv2D(filters=1, kernel_size=3, strides=(1,1), padding='same', name='conv10')(conv9)

  model = Model(inputs=i, outputs=o)
  return model

if __name__ == "__main__":
  model = build(128,128,3)
  model.summary()