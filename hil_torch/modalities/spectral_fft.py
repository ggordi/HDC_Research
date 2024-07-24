# encoding images using sparse spectral features, computed via fast fourier transform

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


# linearly encode bins for the magnitudes of the frequencies in the images
def encode_bins(num_bins):
    f_train = np.abs(np.fft.fft2(train_images))
    min_magnitude = np.min(f_train)
    max_magnitude = np.max(f_train)
    bins = np.linspace(min_magnitude, max_magnitude, num_bins)
    print(bins)


def highpass_filter(matrix, drop_pctg):
    # fft the image and center zero frequency
    f = np.fft.fft2(matrix)
    f = np.fft.fftshift(f)

    # compress the image
    rows, cols = f.shape
    drop_count = int(drop_pctg * min(rows, cols))  # calculate number of frequencies to drop
    # drop frequencies nearest to the center, which are the lowest
    f[int(rows / 2 - drop_count):int(rows / 2 + drop_count), int(cols / 2 - drop_count):int(cols / 2 + drop_count)] = 0

    # bin the frequencies


encode_bins(32)

# next steps:
# - going to encode based on the magnitudes of each frequency (that's the feature the model should be learning)
# - use np.abs to get the magnitude
# - then, bin each magnitude to a certain bin vector (encode these bins, can be linearly/equidistant, use
#   np.digitize like in quantization file)
