# encoding images using a distributed quantization of pixel intensities

import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import vector as vec
from bases import output_vecs


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
    num_bins = 30
    bin_edges = np.unique(np.percentile(values, np.linspace(0, 100, num_bins + 1)))  # handle excess zeros
    print([int(x) for x in list(bin_edges)])
    # visualize binning
    plt.hist(values, bins=bin_edges, color='blue', edgecolor='black')
    plt.xlabel('pixel intensity')
    plt.ylabel('number of occurrences')
    plt.title('histogram with binning')
    plt.show()


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

edges = [0, 11, 42, 69, 93, 116, 136, 155, 171, 185, 197, 207, 217, 226, 237, 255]  # calculate_bins() num_bins = 30

# encode bins so the central vector represents the mean, increase hamming distance to mean vector as move over bins
# use something like (difference in pixels) * 20 count bit changes per bin
# so, going from bin 0 to bin 1, 0 to 11, will change 11 - 0 = 11 * 20 = 220 bits
# --> 255 * 20 = 5100 total changes (orthogonality)
# notes about changing digits:
# - for preceding vectors, start from the beginning of the vector (flip bits starting at 0)
# - for ensuing vectors, start flipping digits from the end of the vector (starting at 10000)

bin_vecs = {3: vec.Vector()}  # represents the mean, bin 3 (mean intensity 73 maps to bin 4)

cur = 3
idx = 0

print(bin_vecs[cur])

for i in range(2, -1, -1):  # encode vectors for the bins preceding center
    dif = (edges[cur] - edges[i]) * 20  # how many bits to flip
    cur_vec = bin_vecs[cur]
    next_bits = cur_vec.bits.clone()  # type = tensor
    for j in range(idx, idx + dif + 1):
        next_bits[j] = next_bits[j] ^ 1  # xor with 0 to flip digit
    idx += dif
    bin_vecs[cur-1] = vec.Vector(next_bits)
    cur -= 1

cur = 3
idx = 9999
for i in range(4, 10):  # encode vectors for the bins following center
    dif = (edges[i] - edges[cur]) * 20
    cur_vec = bin_vecs[cur]
    next_bits = cur_vec.bits.clone()
    for j in range(idx, idx - dif - 1, -1):
        next_bits[j] = next_bits[j] ^ 1
    idx -= dif
    bin_vecs[cur+1] = vec.Vector(next_bits)
    cur += 1

print(bin_vecs)


for i in range(0, 9):
    print(f'hd vec {i} to {i+1} = {vec.hamming_distance(bin_vecs[i], bin_vecs[i+1])}')



