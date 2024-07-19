# encoding images using a distributed quantization of pixel intensities

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import vector as vec
from bases import output_vecs

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

bins = [0, 3, 17, 36, 53, 69, 83, 98, 111, 124, 136, 148, 158, 168, 177, 185, 192, 199, 205, 211, 217, 222,
        228, 234, 244, 255]  # from running calculate_bins()
bin_vectors = [vec.Vector() for x in bins]


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
    return total / (784 * 10000)


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


# encode an image (matrix of intensities) using the bins defined above
# question: should i be binding the bin vectors to location (row/col) vectors as well?
def encode_bins(matrix):
    pixel_vecs = []
    for row in matrix:
        for intensity in row:
            # search for the correct bin index
            idx = 1
            for i in range(0, len(bins)):  # change to more efficient, nonlinear search later
                if intensity <= bins[idx]:
                    break
                else:
                    idx = i
            pixel_vecs.append(bin_vectors[idx])
    return vec.consensus_sum(pixel_vecs)


# train and test using quantization
def train_test(num_examples):
    # train
    class_vecs = {i: [] for i in range(0, 10)}

    for i in range(num_examples):
        print(f'encoding img {i}')
        class_vecs[train_labels[i]].append(encode_bins(train_images[i]))

    sums = []
    for i in range(10):
        cons = vec.consensus_sum(class_vecs[i])
        if cons is not None:
            sums.append(cons)

    bound_vecs = []
    count = 0
    for sum_vec in sums:
        bound_vecs.append(vec.xor(sum_vec, output_vecs[count]))
        count += 1

    hil = vec.consensus_sum(bound_vecs)

    # test
    correct = 0
    for i in range(1000):
        print(f'testing img {i}')
        t = encode_bins(test_images[i])
        res = vec.xor(hil, t)

        min_hd = 10000
        prediction = -1
        for j in range(10):
            cur_hd = vec.hamming_distance(res, output_vecs[j])
            if cur_hd < min_hd:
                min_hd = cur_hd
                prediction = j
        expected = test_labels[i]

        if prediction == expected:
            correct += 1

    return correct / 1000


print(train_test(10000))  # achieved 10.4% accuracy when trained on 10,000 examples


# next: use distribution statistics to determine bin edges. then, assign each bin a hypervector.
# encode an image using the corresponding bin hypervector for each pixel. sum into a single representation
# for the image.
# - use narrower bins for intensities that occur more frequently (small variations are captured more precisely)
# - use wider bins for less frequent intensities as they simplify the representation

# question: should this be used in combination with the bases for geographical location? so,
# just replacing the pixel intensity part of the encoding in hil.py, keeping the positional
# things the same?





