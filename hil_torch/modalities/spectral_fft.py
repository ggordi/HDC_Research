# encoding images using sparse spectral features, computed via fast fourier transform

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

img = train_images[0]

img_fft = np.fft.fft2(img)  # compute 2d fft
shifted_fft = np.fft.fftshift(img_fft)  # shift zero to center
mag_spec = np.abs(shifted_fft)  # magnitude spectrum

plt.title('magnitude spectrum')
plt.imshow(mag_spec)
plt.colorbar()
plt.show()


