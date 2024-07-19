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


# resulting bin edges: [0, 3, 17, 36, 53, 69, 83, 98, 111, 124, 136, 148, 158, 168, 177, 185, 192, 199, 205, 211,
# 217, 222, 228, 234, 244, 255]
def calculate_bins():
    values = np.sort(train_images.flatten())
    num_bins = 50
    bin_edges = np.unique(np.percentile(values, np.linspace(0, 100, num_bins + 1)))  # handle excess zeros
    print([int(x) for x in list(bin_edges)])
    # visualize binning
    plt.hist(values, bins=bin_edges, color='blue', edgecolor='black')
    plt.xlabel('pixel intensity')
    plt.ylabel('number of occurrences')
    plt.title('histogram with binning')
    plt.show()


# next: use distribution statistics to determine bin edges. then, assign each bin a hypervector.
# encode an image using the corresponding bin hypervector for each pixel. sum into a single representation
# for the image.
# - use narrower bins for intensities that occur more frequently (small variations are captured more precisely)
# - use wider bins for less frequent intensities as they simplify the representation

# question: should this be used in combination with the bases for geographical location? so,
# just replacing the pixel intensity part of the encoding in hil.py, keeping the positional
# things the same?


