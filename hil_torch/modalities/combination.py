# combining modalities to analyze which ones improve or hurt the performance of the image classification model
# - spatial intensity = SI
# - histogram of pixel intensities = HIST
# - quantization of pixel intensities = QT
# - sparse spectral features, fft = SFFT
# - sparse spectral features, dct = SDCT
# - convolutional features = CONV
# process: for each image, encode with the desired modalities and each vector result to a symbolic vector for the
#          modality, and sum into a single representation for the image

import tensorflow as tf
import vector as vec
from bases import output_vecs

from spatial_intensity import encode_si
from quantization import encode_qt
from histogram import encode_hist
from spectral_dct import encode_dct
from spectral_fft import encode_fft

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

encoders = {'SI': encode_si, 'QT': encode_qt, 'HIST': encode_hist, 'DCT': encode_dct, 'FFT': encode_fft}
modality_vecs = {x: vec.Vector() for x in ['SI', 'QT', 'HIST', 'DCT', 'FFT']}


# will soon change this to use binomial expansion
def combined_hil(modalities, num_examples=1000):
    class_vecs = {i: [] for i in range(10)}
    for i in range(num_examples):
        print(f'training img {i+1}')
        img_vecs = []
        for m in modalities:
            img_vecs.append(vec.xor(modality_vecs[m], encoders[m](train_images[i])))
        class_vecs[train_labels[i]].append(vec.consensus_sum(img_vecs))

    sums = [vec.consensus_sum(class_vecs[i]) for i in range(10)]
    bound_vecs = [vec.xor(output_vecs[i], sums[i]) for i in range(10)]

    return vec.consensus_sum(bound_vecs)


def test_combined(modalities, hil, num_examples=1000):  # specify which modalities were used in the HIL encoding
    correct = 0

    for i in range(num_examples):
        img_vecs = []
        for m in modalities:  # image = sum( encode(img) xor modality_vec ) for each encoding function
            img_vecs.append(vec.xor(encoders[m](test_images[i]), modality_vecs[m]))
        img = vec.consensus_sum(img_vecs)
        res = vec.xor(img, hil)

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

    return correct / num_examples


mods = ['SI', 'DCT']
comb_hil = combined_hil(mods)
print(f'{mods} = {test_combined(mods, comb_hil)}')

# SI + DCT = 44%
