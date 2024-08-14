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


def encode_si_lp(image):
    lp_image = cv2.logPolar(image, (13, 13), 28/np.log(28), cv2.WARP_FILL_OUTLIERS)
    pixels = []
    for row in range(0, len(lp_image)):
        for col in range(0, len(lp_image[row])):
            pixel_vec = intensity_vecs[lp_image[row][col]]
            pixel_vec = vec.xor(pixel_vec, row_vecs[row])
            pixel_vec = vec.xor(pixel_vec, col_vecs[col])
            pixels.append(pixel_vec)
    return vec.consensus_sum(pixels)


def train_test(num_examples=1000, test_count=1000):
    # train
    class_vecs = {i: [] for i in range(10)}
    itr = iter(train_loader)
    for i in range(num_examples):
        print(f'train #{i+1}')
        obj = next(itr)
        img = obj[0].numpy().squeeze()
        label = obj[1].item()
        class_vecs[label].append(encode_si_lp(img))

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
        print(f'test #{i+1}')
        obj = next(itr2)
        img = obj[0].numpy().squeeze()
        label = obj[1].item()
        t = encode_si_lp(img)
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

print(train_test())

# 54.3%, 49.7%, 47.3%, 48.8% with 1000 training examples
