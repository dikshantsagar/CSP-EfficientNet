from cspefficientnet import EfficientNetB7

import cspefficientnet
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

#from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("best_model_csp.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
model = EfficientNetB7(weights='best_model_csp.hdf5', classes=10)
optim = tf.keras.optimizers.RMSprop()
model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs_csp', histogram_freq=1), checkpoint]
model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), callbacks=callbacks)
