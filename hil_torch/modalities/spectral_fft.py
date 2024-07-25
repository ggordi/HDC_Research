# encoding images using sparse spectral features, computed via fast fourier transform

import numpy as np
import tensorflow as tf
import vector as vec
from bases import row_vecs, col_vecs, output_vecs
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


# calculate bin edges
# fix this --> linear spacing is not going to work
def calculate_edges(num_bins):
    # create bins for the frequencies shown in the dataset
    f_train = np.abs(np.fft.fft2(train_images)).flatten()
    return list(np.percentile(np.sort(f_train), np.linspace(0, 100, num_bins + 1)))


# linearly encode bins for the magnitudes of the frequencies in the images
# fix this --> linear encoding is not going to work
def encode_bins(num_bins):
    bins = [vec.Vector()]
    changes = 5000 // num_bins
    for i in range(0, num_bins):
        cur_vec = bins[i]
        next_bits = cur_vec.bits.clone()
        for j in range(changes * i, changes * (i + 1)):
            next_bits[j] ^= 1
            bins.append(vec.Vector(next_bits))
    return bins


def encode_fft(matrix, drop_pct=0.9):
    # fft the image and center zero frequency
    f = np.fft.fft2(matrix)
    f = np.fft.fftshift(f)

    # compress the image
    rows, cols = f.shape
    drop_count = int(drop_pct * min(rows, cols))  # calculate number of frequencies to drop
    # drop frequencies nearest to the center, which are the lowest
    f[int(rows / 2 - drop_count):int(rows / 2 + drop_count), int(cols / 2 - drop_count):int(cols / 2 + drop_count)] = 0

    # bin the frequencies and hypervector encoding
    f = np.digitize(np.abs(f), edges)
    freq_vecs = []
    for row in range(0, len(f)):
        for col in range(0, len(f[row])):
            if f[row][col] != 0:
                freq_vec = bin_vecs[f[row][col]]
                freq_vec = vec.xor(vec.xor(freq_vec, row_vecs[row]), col_vecs[col])  # bind magnitude to frequency
                freq_vecs.append(freq_vec)

    return vec.consensus_sum(freq_vecs)


# train and test
def train_test(num_examples):
    # train
    class_vecs = {i: [] for i in range(10)}

    for i in range(num_examples):
        print(f'encoding image {i + 1}')
        class_vecs[train_labels[i]].append(encode_fft(train_images[i]))

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
        print(f'testing image {i + 1}')
        t = encode_fft(test_images[i])
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


edges = calculate_edges(32)
bin_vecs = encode_bins(32)

print(train_test(1000))

# 9.8% with 48 bins, 1000 training examples
# 11.6% with 32 bins, 1000 training examples
# 11.6% with 20 bins, 1000 training examples
# 9.9% with 12 bins, 1000 training examples
