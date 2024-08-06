# note: need to update 3x3 blocking to include overlapping, as of now it does not.

# encoding images using sparse spectral features, using discrete cosine transform
import numpy as np
import tensorflow as tf
from scipy.fftpack import dctn, idctn
import vector as vec
import matplotlib.pyplot as plt
from bases import output_vecs, pos_vecs

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


def calculate_edges(num_bins):
    d_train = []
    for i in range(0, 26, 3):
        for j in range(0, 26, 3):
            block = train_images[:, i:i + 3, j:j + 3]
            d_train.extend(np.abs(dctn(block)).flatten())
    return np.percentile(np.sort(d_train), np.linspace(0, 100, num_bins + 1))


# adjust so hamming distance is proportionate to difference between bin frequency values
def encode_bins(num_bins):
    bins = [vec.Vector()]
    changes = 5000 // num_bins  # 5000 represents randomness when vectors are of length 10000
    for i in range(0, num_bins):
        cur_vec = bins[i]
        next_bits = cur_vec.bits.clone()
        for j in range(changes * i, changes * (i + 1)):
            next_bits[j] ^= 1
            bins.append(vec.Vector(next_bits))
    return bins


def encode_dct(matrix):
    # dct the image in 3x3 pixel blocks
    blocks = []
    for i in range(0, 26, 3):
        for j in range(0, 26, 3):
            blocks.append(matrix[i:i + 3, j:j + 3])
    dct_blocks = [dctn(block) for block in blocks]

    # preserve only highest single frequency out of the 9 in each block, and bin them
    freq = []
    for block in dct_blocks:
        freq.append(np.abs(np.max(block)))
    block_bins = np.digitize(freq, edges)

    # the index of the frequency in the list indicates its position in the image
    freq_vecs = []
    for i in range(0, len(block_bins)):
        freq_vecs.append(vec.xor(bin_vecs[block_bins[i]], pos_vecs[i]))  # bind the bin to its location

    return vec.consensus_sum(freq_vecs)


# train and test
def train_test(num_examples):
    # train
    class_vecs = {i: [] for i in range(10)}

    for i in range(num_examples):
        print(f'encoding image {i + 1}')
        class_vecs[train_labels[i]].append(encode_dct(train_images[i]))

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
        t = encode_dct(test_images[i])
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


edges = calculate_edges(150)
bin_vecs = encode_bins(150)

# print(train_test(1000))

# 10.7% accuracy with 150 bins, 1000 examples
