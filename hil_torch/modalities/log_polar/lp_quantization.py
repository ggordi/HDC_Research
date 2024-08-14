# encoding images via quantization of pixel intensities - log polar

import torch
import torchvision.datasets
import torchvision.transforms as transforms

import cv2
import numpy as np

import vector as vec
from intensities import intensity_vecs
from bases import row_vecs, col_vecs, output_vecs

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).int())  # scale to [0, 255]
])

train_set = torchvision.datasets.FashionMNIST("../data", download=True, transform=tfm)
test_set = torchvision.datasets.FashionMNIST("../data", download=True, train=False, transform=tfm)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)


# mean intensity value over first 10000 images ~= 106.146
def get_mean():
    hist = {i: 0 for i in range(0, 256)}
    count = 10000
    itr = iter(train_loader)
    for i in range(count):
        image = next(itr)[0].numpy().squeeze()
        lp_image = cv2.logPolar(image, (13, 13), 28 / np.log(28), cv2.WARP_FILL_OUTLIERS)
        for row in lp_image:
            for pixel in row:
                hist[pixel] += 1
    total = 0
    for key, value in hist.items():
        total += key * value
    return total / (784 * 10000)


def get_edges(num_bins=48):
    vals = []
    for batch in train_loader:
        image = batch[0].numpy().squeeze()
        lp_image = cv2.logPolar(image, (13, 13), 28 / np.log(28), cv2.WARP_FILL_OUTLIERS)
        vals.extend(lp_image.flatten())
    vals = np.sort(vals)
    # excess 0s decreases total number of bins
    bin_edges = np.unique(np.percentile(vals, np.linspace(0, 100, num_bins + 1)))
    return list(bin_edges.astype(int))


def encode_bins():
    bin_vecs = {8: vec.Vector()}  # represents the mean, bin 8

    indexes = np.arange(10000)  # using vectors of length 10000

    cur = 8
    for i in range(cur - 1, -1, -1):  # encode vectors for the bins preceding center
        dif = (edges[cur] - edges[i]) * 20  # how many bits to flip
        cur_vec = bin_vecs[cur]
        next_bits = cur_vec.bits.clone()
        to_change = np.random.choice(indexes, size=dif, replace=False)  # which bits to flip
        for x in to_change:
            next_bits[x] ^= 1  # xor with 1 to flip digit
        indexes = np.setdiff1d(indexes, to_change)
        bin_vecs[cur - 1] = vec.Vector(next_bits)
        cur -= 1

    cur = 8
    for i in range(cur + 1, len(edges)):  # encode vectors for the bins following center
        dif = (edges[i] - edges[cur]) * 20
        cur_vec = bin_vecs[cur]
        next_bits = cur_vec.bits.clone()
        to_change = np.random.choice(indexes, size=dif, replace=False)
        for x in to_change:
            next_bits[x] ^= 1  # xor with 1 to flip digit
        bin_vecs[cur + 1] = vec.Vector(next_bits)
        cur += 1

    return bin_vecs


def encode_qt_lp(image):
    lp_image = cv2.logPolar(image, (13, 13), 28 / np.log(28), cv2.WARP_FILL_OUTLIERS)
    image_vecs = []
    for row in range(28):
        for col in range(28):
            cur = bins[np.digitize(lp_image[row][col], edges) - 1]
            image_vecs.append(vec.xor(vec.xor(cur, row_vecs[row]), col_vecs[col]))
    return vec.consensus_sum(image_vecs)


def train_test(num_examples=1000, test_count=1000):
    # train
    class_vecs = {i: [] for i in range(10)}
    itr = iter(train_loader)
    for i in range(num_examples):
        print(f'train #{i + 1}')
        obj = next(itr)
        img = obj[0].numpy().squeeze()
        label = obj[1].item()
        class_vecs[label].append(encode_qt_lp(img))

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
        print(f'test #{i + 1}')
        obj = next(itr2)
        img = obj[0].numpy().squeeze()
        label = obj[1].item()
        t = encode_qt_lp(img)
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


edges = [0, 4, 24, 41, 55, 68, 80, 91, 102, 112, 121, 131, 139, 147, 155, 161, 168, 174, 180, 185, 190, 195, 199, 203,
         208, 212, 216, 219, 223, 227, 231, 236, 244, 255]  # get_edges()
bins = encode_bins()

print(train_test())

# results = 39.9%, 48.8%, 41.1% with 1000 training examples
