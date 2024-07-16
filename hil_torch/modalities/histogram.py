import vector as vec
from bases import output_vecs
from intensities import intensity_vecs
import numpy as np


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

count_vecs = create_count_vecs(28 * 28)


# create histogram dictionary for entered matrix of intensity values
def create_histogram(matrix, width, height):
    hist = {}
    for row in range(0, width):
        for col in range(0, height):
            pix = matrix[row][col]
            hist[pix] = hist.get(pix, 0) + 1  # set to 1 if not present
    return hist


# encode an image using the histogram of its pixel intensities
def encode_histogram(matrix, width, height):
    img_hist = create_histogram(matrix, width, height)
    vec_list = []
    for intensity, count in img_hist.items():
        vec_list.append(vec.xor(intensity_vecs[intensity], count_vecs[count]))  # count_vecs created above
    return vec.consensus_sum(vec_list)


# train the model and test it


