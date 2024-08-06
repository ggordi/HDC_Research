# encoding images using sparse spectral features, computed via fast fourier transform

import vector as vec
from bases import row_vecs, col_vecs, output_vecs
import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


drop = 90
fft_min = 2047.2452255411417
fft_max = 150387.0
# fft_min, fft_max need to be updated with any change to drop, edges need to be updated with any change to num_bins
num_bins = 24
edges = [2047.2452255411417, 8228.068341143593, 14408.891456746045, 20589.714572348497, 26770.53768795095,
         32951.360803553405, 39132.183919155854, 45313.0070347583, 51493.83015036076, 57674.653265963214,
         63855.47638156566, 70036.29949716812, 76217.12261277057, 82397.94572837303, 88578.76884397547,
         94759.59195957793, 100940.41507518038, 107121.23819078284, 113302.0613063853, 119482.88442198774,
         125663.70753759019, 131844.53065319263, 138025.3537687951, 144206.17688439754, 150387.0]


def define_bins():
    # calculate the min and max value for the top desired percentage
    mags = []
    for image in train_images:
        mags.extend(np.abs(np.fft.fft2(image)).flatten())
    mags = np.array(mags)
    threshold = np.percentile(mags, drop)
    max_val = np.max(mags)
    print(f' threshold value = {threshold} \n max value = {max_val}')
    # calculate the bin edges according to the results above
    bin_edges = np.linspace(fft_min, fft_max, num_bins + 1)
    print(list(bin_edges))


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


def encode_fft(image):
    # perform fft on the image, calculate threshold
    fft_res = np.abs(np.fft.fft2(image))
    threshold = np.percentile(np.sort(fft_res.flatten()), drop)

    # replace dropped frequencies with 0s
    for row in range(len(fft_res)):
        for col in range(len(fft_res[row])):
            if fft_res[row][col] < threshold:
                fft_res[row][col] = 0

    # bin and encode the remaining values
    fft_res = np.digitize(fft_res, edges)
    image_vecs = []
    for row in range(len(fft_res)):
        for col in range(len(fft_res[row])):
            if fft_res[row][col] > 0:
                cur = vec.xor(vec.xor(bin_vecs[fft_res[row][col]], row_vecs[row]), col_vecs[col])
                image_vecs.append(cur)

    return vec.consensus_sum(image_vecs)


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


bin_vecs = encode_bins()

# print(define_bins())

print(train_test(10000))

# results:
# 48.2% with 1000 examples, 24 bins
# 49.4% with 10000 examples, 24 bins
# 47.4% with 1000 examples, 32 bins
# 48.3% with 10000 examples, 32 bins
# 47.3% with 1000 examples, 48 bins
# 49.7% with 10000 examples, 48 bins
