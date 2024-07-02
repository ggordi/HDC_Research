# linear encoding for pixel intensities

from vector import Vector
import numpy as np
import random

start = Vector()  # orthogonal vectors for 0 and 255
end = Vector()
intensity_vecs = [start]

diff = []  # track the indexes of mismatches
for i in range(0, Vector.length):
    if start.bits[i] != end.bits[i]:
        diff.append(i)

changes = int(len(diff) / 255)  # increase distance between vectors linearly
recent = intensity_vecs[0]

# for distributing the leftover indexes
extra = random.sample(range(0, 254), len(diff) - 254 * changes)

for i in range(0, 254):  # only do this for the 254 vectors between start and end
    choices = None

    if i in extra:
        choices = np.random.choice(diff, changes + 1, False)
    else:
        choices = np.random.choice(diff, changes, False)
    diff = np.setdiff1d(diff, choices).tolist()
    new_vec = Vector(recent.bits.clone())

    for x in choices:
        new_vec.bits[x] = end.bits[x]
    intensity_vecs.append(new_vec)
    recent = intensity_vecs[i+1]

intensity_vecs.append(end)