import torch
import random

class Vector:
    length = 10000

    def __init__(self, bits=None):
        if bits is None:
            self.bits = torch.randint(0, 2, (self.length,), dtype=torch.uint8, device='cuda')
        else:
            self.bits = bits.clone().detach().to('cuda', dtype=torch.uint8)

def hamming_distance(v1: Vector, v2: Vector):
    # converts a boolean tensor to an int tensor, returning its sum
    return torch.sum(torch.tensor(v1.bits != v2.bits).to(torch.uint8)).item()


def xor(v1: Vector, v2: Vector):
    res_bits = torch.bitwise_xor(v1.bits, v2.bits)
    return Vector(res_bits)

def consensus_sum(vectors: list[Vector]):
    if len(vectors) == 0:
        return None

    # stack the bits into a single tensor, sum to get ones count
    bits = torch.stack([vec.bits for vec in vectors])
    ones_count = torch.sum(bits, dim=0)
    majority = len(vectors) / 2

    cons_bits = torch.zeros(Vector.length, dtype=torch.uint8, device='cuda')
    cons_bits[ones_count > majority] = 1

    ties = (ones_count == majority)
    cons_bits[ties] = torch.randint(0, 2, (torch.sum(torch.tensor(ties).to(torch.uint8)).item(),),
                                    dtype=torch.uint8, device='cuda')

    return Vector(cons_bits)




