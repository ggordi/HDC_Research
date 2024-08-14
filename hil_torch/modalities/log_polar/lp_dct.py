# encoding images via sparse spectral features from discrete cosine transform - log polar
import torch
import torchvision.datasets
import torchvision.transforms as transforms

import cv2
import numpy as np
from scipy.fft import dctn

import vector as vec
from bases import pos_vecs, output_vecs

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).int())  # scale to [0, 255]
])

train_set = torchvision.datasets.FashionMNIST("../data", download=True, transform=tfm)
test_set = torchvision.datasets.FashionMNIST("../data", download=True, train=False, transform=tfm)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)


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


def get_edges(start, stop):
    return list(np.linspace(start, stop, num_bins + 1))


def encode_dct_lp(image):
    lp_image = cv2.logPolar(image, (13, 13), 28 / np.log(28), cv2.WARP_FILL_OUTLIERS)
    # process the image in 3x3 blocks
    blocks = []
    for i in range(0, 26, 3):
        for j in range(0, 26, 3):
            blocks.append(lp_image[i:i + 3, j:j + 3])

    # dct each block and retain the highest frequency, bin the values
    highest_fre = []
    for block in blocks:
        dct_res = dctn(block, norm='ortho')
        highest_fre.append(max(dct_res.flatten()))
    fre_bins = np.digitize(highest_fre, edges)

    # bind the bin vectors to their position
    fre_vecs = []
    for i, val in enumerate(fre_bins):
        fre_vecs.append(vec.xor(pos_vecs[i], bin_vecs[val]))

    return vec.consensus_sum(fre_vecs)


def train_test(num_examples=1000, test_count=1000):
    # train
    class_vecs = {i: [] for i in range(10)}
    itr = iter(train_loader)
    for i in range(num_examples):
        # print(f'train #{i + 1}')
        obj = next(itr)
        img = obj[0].numpy().squeeze()
        label = obj[1].item()
        class_vecs[label].append(encode_dct_lp(img))

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
    itr2 = iter(test_loader)
    for i in range(test_count):
        # print(f'test #{i + 1}')
        obj = next(itr2)
        img = obj[0].numpy().squeeze()
        label = obj[1].item()
        t = encode_dct_lp(img)
        res = vec.xor(hil, t)

        min_hd = 10000
        prediction = -1
        for j in range(10):
            cur_hd = vec.hamming_distance(res, output_vecs[j])
            if cur_hd < min_hd:
                min_hd = cur_hd
                prediction = j

        if prediction == label:
            correct += 1

    return correct / test_count


num_bins = 32
edges = get_edges(0, 800)  # max frequency values will range from 0 to less than 800
bin_vecs = encode_bins()

for i in range(10):
    print(train_test())


# results averaging 48.4% with 1000 training examples
