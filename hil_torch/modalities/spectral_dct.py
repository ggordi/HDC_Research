# encoding images using sparse spectral features, computed via discrete cosine transform

import tensorflow as tf
from scipy.fft import dctn
import numpy as np

import vector as vec
from bases import output_vecs, pos_vecs

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

def encode_bins():
    bin_vectors = [vec.Vector()]
    indexes = np.arange(10000)  # using vectors of length 10000

    for i in range(num_bins):
        bin_bits = bin_vectors[-1].bits.clone()
        to_change = np.random.choice(indexes, size=10000 // num_bins, replace=False)
        for idx in to_change:
            bin_bits[idx] ^= 1
        indexes = np.setdiff1d(indexes, to_change)
        bin_vectors.append(vec.Vector(bin_bits))

    return bin_vectors


def get_edges(start, stop):
    return list(np.linspace(start, stop, num_bins + 1))


def encode_dct(image):
    # process the image in 3x3 blocks
    blocks = []
    for i in range(0, 26, 3):
        for j in range(0, 26, 3):
            blocks.append(image[i:i + 3, j:j + 3])

    # dct each block and retain the highest frequency, bin the values
    highest_fre = []
    for block in blocks:
        dct_res = dctn(block, norm='ortho')
        highest_fre.append(max(dct_res.flatten()))
    fre_bins = np.digitize(highest_fre, edges)

    # bind the bin vectors to their position
    fre_vecs = []
    for i, val in enumerate(fre_bins):
        fre_vecs.append(vec.xor(pos_vecs[i], bin_vecs[val]))

    return vec.consensus_sum(fre_vecs)


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


num_bins = 32
edges = get_edges(0, 800)  # max frequency values will range from 0 to less than 800
bin_vecs = encode_bins()

print(train_test(10000))
