# encoding images using a distributed quantization of pixel intensities

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# calculated mean pixel intensity over the first 10000 images = 73.009
def calculate_mean():
    hist = {i: 0 for i in range(0, 256)}
    for image in train_images[:10000]:
        for row in image:
            for pixel in row:
                hist[pixel] += 1

    plt.bar(list(hist.keys()), list(hist.values()), color='blue', width=0.4)
    plt.xlabel('pixel intensity')
    plt.ylabel('number of occurrences')
    plt.yscale('log')
    plt.show()

    total = 0
    for key, value in hist.items():
        total += key * value
    return total/(784*10000)


# next: use distribution statistics to determine bin edges. then, assign each bin a hypervector.
# encode an image using the corresponding bin hypervector for each pixel. sum into a single representation
# for the image.

