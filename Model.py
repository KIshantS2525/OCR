import os
import tarfile
import zipfile
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from Dataset import train_ds, validation_ds, test_ds, num_to_chars, char_to_num

np.random.seed(42)
tf.random.set_seed(42)

batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

class CTCLayer(keras.layers.Layer):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.loss_fn = keras.backend.ctc_batch_cost

  def call(self, y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = self.loss_fn(y_true, y_pred, input_length, label_length)
    self.add_loss(loss)


    return y_pred

def build_model():
  input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
  labels = keras.layers.Input(name="label", shape=(None,))

  # first conv block
  x = keras.layers.Conv2D(
    filters=32, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal',
    name="Conv1")(input_img)

  x = keras.layers.Conv2D(
    filters=32, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal',
    name="Conv2")(x)


  x = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1")(x)

  x = keras.layers.Conv2D(
    filters=64, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal',
    name="Conv3")(x)
  
  # Second conv block
  x = keras.layers.Conv2D(
    filters=128, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal',
    name="Conv4")(x)


  x = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2")(x)


  new_shape = ((image_width // 4), (image_height // 4) * 128)
  x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
  x = keras.layers.Dense(64, activation="relu", kernel_initializer='he_normal', name="dense1")(x)
  x = keras.layers.Dense(128, activation="relu", kernel_initializer='he_normal', name="dense2")(x)
  x = keras.layers.Dropout(0.4)(x)

  # RNN
  x = keras.layers.Bidirectional(
      keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
  x = keras.layers.Bidirectional(
      keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)


  x = keras.layers.Dense(
    len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense3")(x)
  # Add CTC layer for calculating CTC Loss at each step.
  output = CTCLayer(name="ctc_loss")(labels, x)

  # Define the model.
  model = keras.models.Model(
      inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")


  # Compile the model and return
  model.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-3), loss=None)
  return model


model = build_model()
model.summary()