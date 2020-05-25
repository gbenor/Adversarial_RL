import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model
from tensorflow.keras.models  import model_from_json

# training params
BATCH_SIZE = 128
MAXEPOCHES = 12
LEARNING_RATE = 0.1
LR_DECAY = 1e-6
LR_DROP = 20

def lr_scheduler(epoch):
    return LEARNING_RATE * (0.5 ** (epoch // LR_DROP))

class ConvMinst():
    ''' Build a simple MNIST classification CNN
        The network takes ~3 minutes to train on a normal laptop and reaches roughly 97% of accuracy
        Model structure: Conv, Conv, Max pooling, Dropout, Dense, Dense
    '''
    def __init__(self, fname, train_img, train_labels, test_img, test_labels):
        fname = Path(fname)
        self._fname_json = fname.parent / f"{fname.stem}.json"
        self._fname_weights = fname.parent / f"{fname.stem}.h5"

        if self.model_exists():
            self.load()
        else:
            self.fit(train_img, train_labels, test_img, test_labels)
        score = self._model.evaluate(test_img, test_labels, verbose=0)
        print('ConvMinst Test accuracy:', score[1])
        assert  score[1]>0.9, f"The model is not trained well. accuracy={score[1]}"

    def fit(self, train_img, train_labels, test_img, test_labels):
        activation = 'relu'
        num_classes = 10
        # input image dimensions
        img_rows, img_cols, img_colors = 28, 28, 1

        model = keras.Sequential()
        model.add(layers.Conv2D(8, kernel_size=(3, 3), input_shape=(img_rows, img_cols, img_colors), activation=activation))
        model.add(layers.Conv2D(8, (3, 3), activation=activation))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation=activation))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax', name='y_pred'))
        self._model = model
        self.compile()



        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        self._model .fit(train_img, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=MAXEPOCHES,
                  verbose=1,
                  validation_data=(test_img, test_labels),
                         callbacks=[reduce_lr])
        self.save()

    def compile(self):
        self._model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=[keras.metrics.CategoricalAccuracy()])

        print("finish build and compile conv_minst")


    def model_exists(self):
        return (self._fname_json.exists() and self._fname_weights.exists())

    def save(self):
        # Save JSON config to disk
        with self._fname_json.open("w") as json_file:
            json_file.write(self._model.to_json())
        # Save weights to disk
        self._model.save_weights(str(self._fname_weights))

    def load(self):
        # Reload the model from the 2 files we saved
        with self._fname_json.open("r") as json_file:
            json_config = json_file.read()
        self._model = model_from_json(json_config)
        self._model.load_weights(str(self._fname_weights))
        self.compile()


