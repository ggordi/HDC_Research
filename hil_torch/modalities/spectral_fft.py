# encoding images based on sparse spectral features using the FFT

import numpy as np
import tensorflow as tf
import vector as vec
from bases import output_vecs, row_vecs, col_vecs

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

num_bins = 32
drop = 90


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


def get_start_stop():
    fft_res = np.concatenate([np.fft.fft2(image).flatten() for image in train_images])
    fft_res = np.sort(np.abs(fft_res))
    min_val = np.percentile(fft_res, drop)
    max_val = fft_res[-1]
    print(f'min = {min_val}\nmax = {max_val}')
    # min = 2047.2452255411417
    # max = 150387.0


def encode_fft(image):
    # fft image and calculate threshold
    fft_res = np.abs(np.fft.fft2(image))
    threshold = np.percentile(np.sort(fft_res.flatten()), drop)

    # drop frequencies under threshold
    for i in range(len(image)):
        for j in range(len(image[i])):
            if fft_res[i][j] < threshold:
                fft_res[i][j] = 0

    # bin remaining values
    fft_res = np.digitize(fft_res, edges)
    fre_vecs = []
    for i in range(len(fft_res)):
        for j in range(len(fft_res[i])):
            if fft_res[i][j] > 0:
                cur = bin_vecs[fft_res[i][j]]
                cur = vec.xor(vec.xor(row_vecs[i], cur), col_vecs[j])
                fre_vecs.append(cur)

    return vec.consensus_sum(fre_vecs)


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


edges = get_edges(2047.2452255411417, 150387.0)
bin_vecs = encode_bins()


print(train_test(1000))

# results with 1000 training examples...
# 48.9%, 45.3%, 47.1%, 47.1%, 44.2%, 43%, 48.8%
