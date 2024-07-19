# encoding images using pixel intensity histograms

import vector as vec
from bases import output_vecs
from intensities import intensity_vecs
import numpy as np
import tensorflow as tf


# create semantic count vectors, max count is total number of pixels in the image
def create_count_vecs(max_count):
    vec_list = [vec.Vector()]
    num_changes = 5000 // max_count

    flip_bits = np.random.choice(np.arange(0, 10000), size=5000, replace=False)

    for i in range(0, max_count + 1):
        new_vec = vec.Vector(vec_list[-1].bits.clone())
        for j in range(i * num_changes, (i + 1) * num_changes):
            new_vec.bits[flip_bits[j]] = new_vec.bits[flip_bits[j]] ^ 1
        vec_list.append(new_vec)

    return vec_list


max_count = 28 * 28  # images in the fashion mnist dataset are 28 by 28 pixels
count_vecs = create_count_vecs(28 * 28)


# create histogram for entered image (matrix of intensity values)
def create_histogram(matrix, width, height):
    hist = {}
    for row in range(0, width):
        for col in range(0, height):
            pix = matrix[row][col]
            if pix in hist:
                hist[pix] += 1
            else:
                hist[pix] = 1
    return hist


# encode an image using the histogram of its pixel intensities
def encode_histogram(matrix, width, height):
    img_hist = create_histogram(matrix, width, height)
    vec_list = []
    for intensity, count in img_hist.items():
        vec_list.append(vec.xor(intensity_vecs[intensity], count_vecs[count]))  # count_vecs created above
    return vec.consensus_sum(vec_list)


# train and test the model
def train_test(num_examples):
    # train
    class_vecs = {i: [] for i in range(10)}

    for i in range(num_examples):
        class_vecs[train_labels[i]].append(encode_histogram(train_images[i], 28, 28))

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
        t = encode_histogram(test_images[i], 28, 28)
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


# train the model and test it
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

print(train_test(50000))

# achieved 13.6% accuracy when trained on 10,000 examples and 35.8% accuracy when trained on 50,000 examples
# in both cases, 1000 images from the test set were used
