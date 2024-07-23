# encoding images using sparse spectral features, computed via fast fourier transform

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

for i in range(0, 4):
    img = train_images[i * 46]
    img_fft = np.fft.fft2(img)  # compute 2d fft
    shifted_fft = np.fft.fftshift(img_fft)  # shift zero to center
    mag_spec = np.abs(shifted_fft)  # magnitude spectrum

    mag_spec = np.log1p(mag_spec)  # log scale for better visualization
    plt.subplot(1, 4, i+1)
    plt.imshow(mag_spec)
    plt.colorbar()

plt.show()
