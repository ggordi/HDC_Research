# hil for a classification model for the fashion mnist dataset

import tensorflow as tf
import vector as vec
from intensities import intensity_vecs
from positions import row_vecs, col_vecs
from outputs import output_vecs
import time


def encode(matrix):
    pixels = []
    for row in range(0, len(matrix)):
        for col in range(0, len(matrix[0])):
            pixel_vec = intensity_vecs[matrix[row][col]]
            pixel_vec = vec.xor(pixel_vec, row_vecs[row])
            pixel_vec = vec.xor(pixel_vec, col_vecs[col])
            pixels.append(pixel_vec)
    return vec.consensus_sum(pixels)


# intakes the resultant vector from t XOR hil, returns the predicted output class via minimizing hamming distance
def predict(vector):
    min_hd = 10000
    prediction = -1
    for i in range(10):
        cur_hd = vec.hamming_distance(vector, output_vecs[i])
        print(f"hd to class {i}: {cur_hd}")
        if cur_hd < min_hd:
            min_hd = cur_hd
            prediction = i
    return prediction


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

start_time = time.time()

class_vecs = {i: [] for i in range(10)}

for i in range(1000):  # append each encoded image to the corresponding class list
    print(f'current i: {i}')
    img = encode(train_images[i])
    class_vecs[train_labels[i]].append(img)

sums = []
for i in range(10):  # create a summed vector representation for each class
    sums.append(vec.consensus_sum(class_vecs[i]))

bound_vecs = []
for i in range(10):  # bind each summed representation
    bound_vecs.append(vec.xor(sums[i], output_vecs[i]))

hil = vec.consensus_sum(bound_vecs)  # consensus sum the bound representations into the hil

# testing
correct = 0
for i in range(100):
    t = encode(test_images[i])
    res = vec.xor(hil, t)

    pre = predict(res)
    expected = test_labels[i]

    if pre == expected:
        correct += 1
    print(f"expected: {expected}, predicted: {pre}")

print(f"correct/total = {correct}/100 = {correct / 100:.2f}")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# result from testing with 500 trained examples:
# correct/total = 48/100 = 0.48
# Elapsed time: 869.42 seconds

# result from testing with 1000 trained examples:
# correct/total = 52/100 = 0.52
# Elapsed time: 1562.24 seconds

# second testing phase... (after adjusting intensity encoding)

# result from testing with 500 trained examples:
# correct/total = 55/100 = 0.55
# Elapsed time: 860.27 seconds

# result from testing with 1000 trained examples
# correct/total = 51/100 = 0.51
# Elapsed time: 1550.51 seconds

# third testing phase... (after linearly encoding the positional vectors)

# 500 training examples
# correct/total = 51/100 = 0.51
# Elapsed time: 864.59 seconds

# 1000 training examples
# correct/total = 48/100 = 0.48
# Elapsed time: 1558.46 seconds
