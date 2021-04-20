import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

x0 = np.loadtxt("/home/mcorsaro/Desktop/0.txt")
x1 = np.loadtxt("/home/mcorsaro/Desktop/1.txt")

latent_dim = 2
original_dim = 6

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            #layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(original_dim, activation='sigmoid')
            #layers.Reshape((6))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def compress(self, x):
        encoded = self.encoder(x)
        return encoded

    def inverseTransform(self, y):
        decoded = self.decoder(y)
        return decoded

sess = tf.Session()

with sess.as_default():

    autoencoder0 = Autoencoder(latent_dim)
    autoencoder1 = Autoencoder(latent_dim)

    autoencoder0.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder1.compile(optimizer='adam', loss=losses.MeanSquaredError())

    print("Input", x0.shape, x1.shape)

    autoencoder0.fit(x0, x0,
                    epochs=10,
                    shuffle=True)

    autoencoder1.fit(x1, x1,
                    epochs=10,
                    shuffle=True)
                    #validation_data=(x_test, x_test))

    x01 = autoencoder0.compress(x1).eval()
    x10 = autoencoder1.compress(x0).eval()
    print("Low-D", x01.shape, x10.shape)

    y2 = np.random.random((100, 2))
    y3 = np.random.random((100, 2))

    x2 = autoencoder0.inverseTransform(y2).eval()
    x3 = autoencoder1.inverseTransform(y3).eval()
    print("Reconstructed", x2.shape, x3.shape)

print("done")