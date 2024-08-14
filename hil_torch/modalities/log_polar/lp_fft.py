# encoding images via sparse spectral features from fast fourier transform - log polar
import torch
import torchvision.datasets
import torchvision.transforms as transforms

import cv2
import numpy as np

import vector as vec
from bases import row_vecs, col_vecs, output_vecs

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).int())  # scale to [0, 255]
])

train_set = torchvision.datasets.FashionMNIST("../data", download=True, transform=tfm)
test_set = torchvision.datasets.FashionMNIST("../data", download=True, train=False, transform=tfm)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

num_bins = 32
drop = 90


def get_start_stop():
    vals = []
    for batch in train_loader:
        image = batch[0].numpy().squeeze()
        lp_image = cv2.logPolar(image, (13, 13), 28 / np.log(28), cv2.WARP_FILL_OUTLIERS)
        vals.extend(np.fft.fft2(lp_image).flatten())
    fft_res = np.sort(np.abs(vals))
    min_val = np.percentile(fft_res, drop)
    max_val = fft_res[-1]
    return f'min = {min_val}\nmax = {max_val}'
    # min = 1928.7936011925328
    # max = 153609.0


def get_edges(start, stop):
    return list(np.linspace(start, stop, num_bins + 1))


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


def encode_fft_lp(image):
    lp_image = cv2.logPolar(image, (13, 13), 28 / np.log(28), cv2.WARP_FILL_OUTLIERS)

    # fft image and calculate threshold
    fft_res = np.abs(np.fft.fft2(lp_image))
    threshold = np.percentile(np.sort(fft_res.flatten()), drop)

    # drop frequencies under threshold
    for i in range(len(lp_image)):
        for j in range(len(lp_image[i])):
            if fft_res[i][j] < threshold:
                fft_res[i][j] = 0

    # bin remaining values
    fft_res = np.digitize(fft_res, edges)
    fre_vecs = []
    for i in range(len(fft_res)):
        for j in range(len(fft_res[i])):
            if fft_res[i][j] > 0:
                cur = bin_vecs[fft_res[i][j]]
                cur = vec.xor(vec.xor(row_vecs[i], cur), col_vecs[j])
                fre_vecs.append(cur)

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
        class_vecs[label].append(encode_fft_lp(img))

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
        t = encode_fft_lp(img)
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


edges = get_edges(1928.7936011925328, 153609.0)
bin_vecs = encode_bins()

for i in range(10):
    print('working...')
    print(train_test(1000))


# results averaging around 35% for 1000 training examples
