import tensorflow as tf
import numpy as np
import os
import time

#Tested on Colab with GPU, Tensorflow 2.4.1
#Tested on MacBook Pro M1 with CPU/GPU, Tensorflow 2.4.0

# ***
# Uncomment the next three lines if you need to run this script in a M1 powered computer.
# ***

#from tensorflow.python.compiler.mlcompute import mlcompute
#from tensorflow.python.keras.api._v2 import keras 
#mlcompute.set_mlc_device(device_name='gpu')

#Load the dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

#Add a trailing unitary dimension to make a 3D multidimensional array (tensor).
# N x 28 x 28 --> N x 28 x 28 x 1
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#Convert the labels from integers to one-hot encoding.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

LR = 1E-3 
EPOCHS = 10
BATCH_SIZE = 64

def build_model(input_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))

    return model

def train(x_train, y_train, x_test, y_test):
    """
    Train the model given the dataset and the global parameters (LR, EPOCHS and BATCH_SIZE).

    The model is automalically saved after the training.

    """
    model = build_model(x_train.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )
    print(model.summary())

    start_time = time.time()

    model.fit(
        x=x_train.astype(np.float32),
        y=y_train.astype(np.float32),
        epochs=EPOCHS,
        validation_data=(x_test.astype(np.float32), y_test.astype(np.float32)),
        batch_size=BATCH_SIZE,
    )

    end_time = time.time()
    print("Train elapsed time: {} seconds".format(end_time - start_time))

    model.save("fashion_mnist_model.tf", overwrite=True)
    


def test(x_test, y_test):
    """
    Load the saved model and evaluate it against the test set.
    """
    model = tf.keras.models.load_model("./fashion_mnist_model.tf")
    print(model.summary())

    start_time = time.time()

    model.evaluate(x_test, y_test)

    end_time = time.time()
    print("Test elapsed time: {} seconds".format(end_time - start_time))

# Comment/uncomment the following two lines as needed.
train(x_train, y_train, x_test, y_test)
test(x_test, y_test)
