# vector utility class and methods

import numpy as np
import random


class Vector:
    length = 10000

    def __init__(self, bits=None):
        if bits is None:
            self.bits = []
            for i in range(0, self.length):
                self.bits.append(np.random.choice([0, 1]))
        else:
            self.bits = bits


def hamming_distance(v1: Vector, v2: Vector):
    count = 0
    for i in range(0, Vector.length):
        if v1.bits[i] != v2.bits[i]:
            count += 1
    return count


def xor(v1: Vector, v2: Vector):
    x_bits = []
    for i in range(0, Vector.length):
        if v1.bits[i] == v2.bits[i]:
            x_bits.append(0)
        else:
            x_bits.append(1)
    return Vector(x_bits)


def consensus_sum(vectors: list[Vector]):
    count = 0  # count the number of 1s seen in a certain position
    bits = []
    for i in range(0, Vector.length):
        for vector in vectors:
            if vector.bits[i] == 1:
                count += 1
        if count > len(vectors) / 2:
            bits.append(1)
        elif count == len(vectors) / 2:  # there was a tie...
            bits.append(random.choice([0, 1]))
        else:
            bits.append(0)
        count = 0
    return Vector(bits)



