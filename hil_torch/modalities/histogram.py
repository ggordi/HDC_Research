import vector as vec
from bases import output_vecs
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


# create histogram dictionary for entered matrix of intensity values
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


count_vecs = create_count_vecs(28 * 28)

img = [[1, 240, 240], [0, 0, 1], [255, 255, 255]]
print(create_histogram(img, 3, 3))
