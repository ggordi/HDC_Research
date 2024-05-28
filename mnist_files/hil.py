# hil for a classification model for the fashion mnist dataset

import tensorflow as tf
import vector as vec
from intensities import intensity_vecs
from positions import row_vecs, col_vecs
from outputs import output_vecs


def encode(matrix):
    pixels = []
    for row in range(0, len(matrix)):
        for col in range(0, len(matrix[0])):
            pixel_vec = intensity_vecs[matrix[row][col]]
            pixel_vec = vec.xor(pixel_vec, row_vecs[row])
            pixel_vec = vec.xor(pixel_vec, col_vecs[col])
            pixels.append(pixel_vec)
    return vec.consensus_sum(pixels)


def print_hd(vector):
    for i in range(10):
        print(f'[{i}] = {vec.hamming_distance(vector, output_vecs[i])}')


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# for now writing out the encoded vectors explicitly, will soon change to an iterative approach

# class 0 expected for train images: 1, 2, 4, 10, 17, 26, 48
s0 = vec.consensus_sum([encode(train_images[1]), encode(train_images[2]), encode(train_images[4])])
b0 = vec.xor(s0, output_vecs[0])

# class 1: 16, 21, 38, 69, 71, 74, 78, 80, 86, 97, 98
s1 = vec.consensus_sum([encode(train_images[16]), encode(train_images[21]), encode(train_images[38])])
b1 = vec.xor(s1, output_vecs[1])

# class 2: 5, 7, 27, 37, 45, 53, 54, 65, 92
s2 = vec.consensus_sum([encode(train_images[5]), encode(train_images[7]), encode(train_images[27])])
b2 = vec.xor(s2, output_vecs[2])

# class 3: 3, 20, 25, 31, 47, 49, 50, 51, 58, 59, 70, 73, 81, 91, 94
s3 = vec.consensus_sum([encode(train_images[3]), encode(train_images[20]), encode(train_images[25])])
b3 = vec.xor(s3, output_vecs[3])

# class 4: 19, 22, 24, 28, 29, 68, 75, 76, 96
# class 5: 8, 9, 12, 13, 30, 36, 43, 60, 62, 63, 82
# class 6: 18, 32, 33, 39, 40, 55, 56, 72, 77, 95
# class 7: 6, 14, 41, 46, 52, 83, 85, 87
# class 8: 23, 35, 57, 99
# class 9: 0, 11, 15, 42, 44, 79, 84, 88, 89, 90, 93

t = encode(train_images[48])  # expected output class of 0
res = vec.xor(t, b0)
print(vec.hamming_distance(res, output_vecs[0]))
res = vec.xor(t, b1)
print(vec.hamming_distance(res, output_vecs[0]))
res = vec.xor(t, b2)
print(vec.hamming_distance(res, output_vecs[0]))
res = vec.xor(t, b3)
print(vec.hamming_distance(res, output_vecs[0]))
# above, we see the expected minimum hamming distance to only the corresponding output class of 0

hil = vec.consensus_sum([b0, b1, b2, b3])
res = vec.xor(t, hil)
print_hd(res)
# unexpected behavior: seeing a hamming distance of ~4k for any class included in the hil


