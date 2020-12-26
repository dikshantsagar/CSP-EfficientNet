from efficientnet import EfficientNetB7

import efficientnet
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

#from tensorflow.keras.applications import EfficientNetB7


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
model = EfficientNetB7(weights=None, classes=10)
optim = tf.keras.optimizers.RMSprop()
model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)]
model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), callbacks=callbacks)