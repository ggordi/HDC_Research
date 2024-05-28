# linear encoding of pixel intensity vectors

import numpy as np
from vector import Vector

start = Vector()  # orthogonal vectors for 0 and 255
end = Vector()
intensity_vecs = [start]

diff = []  # track the indexes of mismatches
for i in range(0, Vector.length):
    if start.bits[i] != end.bits[i]:
        diff.append(i)

changes = int(len(diff) / 255)  # increase distance between vectors linearly
recent = intensity_vecs[0]

for i in range(0, 254):
    choices = np.random.choice(diff, changes, False)
    diff = np.setdiff1d(diff, choices).tolist()
    new_vec = Vector(recent.bits.copy())
    for x in choices:
        new_vec.bits[x] = end.bits[x]
    intensity_vecs.append(new_vec)
    recent = intensity_vecs[i+1]

intensity_vecs.append(end)

# now evenly distribute the remainder across the vectors to avoid bias...
