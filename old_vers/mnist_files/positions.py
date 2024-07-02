# pixel location bases

from vector import Vector
import random
import numpy as np

# row_vecs = []
# col_vecs = []
# for i in range(0, 28):
#     row_vecs.append(Vector())
#     col_vecs.append(Vector())

# implement a linear encoding to improve performance:

def gen_linear_vecs(num_vecs):
    start = Vector()
    end = Vector()
    vecs = [start]

    diff = [i for i in range(Vector.length) if start.bits[i] != end.bits[i]]
    changes = len(diff) // (num_vecs - 1)
    extra = random.sample(range(num_vecs - 1), len(diff) - changes * (num_vecs - 1))  # for remainder distribution

    for i in range(1, num_vecs - 1):
        if i in extra:
            choices = np.random.choice(diff, changes + 1, replace=False)
        else:
            choices = np.random.choice(diff, changes, replace=False)
        diff = np.setdiff1d(diff, choices).tolist()
        new_bits = vecs[-1].bits.copy()
        for j in choices:
            new_bits[j] = end.bits[j]
        vecs.append(Vector(new_bits))

    vecs.append(end)
    return vecs

row_vecs = gen_linear_vecs(28)
col_vecs = gen_linear_vecs(28)





