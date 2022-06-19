import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import argparse


'''
Sea Sponge Detection Skeleton Class
'''

class SeaSpongeDetection:
    def __init__(self):
        pass

    def train(self):
        model = tf.keras.models.Sequential(
            [tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
             tf.keras.layers.MaxPool2D(2, 2),

             tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
             tf.keras.layers.MaxPool2D(2, 2),

             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
             tf.keras.layers.MaxPool2D(2, 2),

             tf.keras.layers.Flatten(),

             tf.keras.layers.Dense(512, activation='relu'),

             tf.keras.layers.Dense(1, activation='sigmoid')])
        model.compile(loss='binary_crossentropy', optimizer= 'adam',
                      metrics=['accuracy'])

        return model

    
    def predict(self):
        model = self.train()
        train = ImageDataGenerator(rescale=1 / 255)
        validation = ImageDataGenerator(rescale=1 / 255)

        train_dataset = train.flow_from_directory('./train-data',
                                                  target_size=(200, 200),
                                                  batch_size=5,
                                                  class_mode='binary')

        validation_dataset = validation.flow_from_directory(
            "./validation data",
            target_size=(200, 200),
            batch_size=5, class_mode='binary')
        history = model.fit(train_dataset, steps_per_epoch=8, epochs=30, validation_data=validation_dataset)

        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        graph = plt.show()

        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        graph2 = plt.show()

        return graph, graph2

    def evaluate(self):
        model = self.train()

        dir_path = './test-sea sponge'

        for i in os.listdir(dir_path):
            img = image.load_img(dir_path + '//' + i, target_size=(200, 200))
            plt.imshow(img)
            plt.show()
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            value = model.predict(images)
            if value == 1:
                print('It is sea sponge')
            else:
                print('It is not sea sponge')

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="./train-data")

    ap.add_argument("-d", "--dataset", required=True,
                    help= "./test-sea sponge")
    args=vars(ap.parse_args())

    return args


def main():
    sea_sponge = SeaSpongeDetection()
    sea_sponge.train()
    sea_sponge.predict()
    sea_sponge.evaluate()


    pass

if __name__ == '__main__':
    main()
