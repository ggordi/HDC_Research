# using hypervector encodings of convolutional feature signals from a CNN

import vector as vec
import numpy as np
from bases import output_vecs, pos_vecs

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.ToTensor()
device = 'cuda'

batch_size = 10
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

train_set, validation_set = torch.utils.data.random_split(train_set, [50000, 10000])


class ConvoNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3)
        self.pool0 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc0 = nn.Linear(in_features=4096, out_features=1024)
        self.drop0 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.drop1 = nn.Dropout(p=0.3)

        self.out = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.pool0(x)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)

        x = F.relu(self.fc0(x))
        x = self.drop0(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = self.out(x)
        normalized_output = torch.tanh(x)

        return x, normalized_output


def encode_bins(num_bins):
    edges = np.linspace(-1, 1, num_bins)
    bin_vecs = [vec.Vector()]

    num_changes = 5000 // num_bins

    for i in range(num_bins):  # gradually change the vector for the first bin
        new_bits = bin_vecs[-1].bits.clone()
        for j in range(i * num_changes, (i + 1) * num_changes):
            new_bits[j] ^= 1
        bin_vecs.append(vec.Vector(new_bits))

    return bin_vecs, edges


# encodes an image into vector form by using the output layer of the CNN
def encode_nn(img):
    net.eval()
    with torch.no_grad():
        _, normalized_output = net(img.unsqueeze(0))
        digitized = np.digitize(normalized_output.cpu().numpy(), bins=edge_vals)[0]

        print(normalized_output)
        print(digitized)

        vecs = []

        for i in range(len(digitized)):  # bind each signal bin to component position
            vecs.append(vec.xor(bins[digitized[i]], pos_vecs[i]))

        return vec.consensus_sum(vecs)


def train_test_hd(num_examples, train_loader, test_loader):
    # train
    class_vecs = {i: [] for i in range(10)}

    for i, data in enumerate(train_loader):
        print(f'encoding vector #{i + 1}')
        if i == num_examples:
            break
        inputs, labels = data[0].to(device), data[1].to(device)

        for image, label in zip(inputs, labels):
            class_vecs[label.item()].append(encode_nn(image))

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
    total = 0.0
    for i, data in enumerate(test_loader):
        if i == 100:  # testing on 100 batches
            break
        inputs, labels = data[0].to(device), data[1].to(device)

        for image, label in zip(inputs, labels):
            print(f'testing vector #{i + 1}')
            total += 1
            t = encode_nn(image)
            res = vec.xor(hil, t)
            min_hd = 10000
            prediction = -1
            for j in range(10):
                cur_hd = vec.hamming_distance(res, output_vecs[j])
                if cur_hd < min_hd:
                    min_hd = cur_hd
                    prediction = j
            if prediction == label.item():
                correct += 1

    return correct / total


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    net = ConvoNet()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    def train_epoch():
        net.train(True)
        running_loss = 0.0
        running_accuracy = 0.0
        total_samples = 0

        i = 0

        for data in train_loader:
            print(f'training data #{i + 1} out of {len(train_loader)}')
            i += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs, _ = net(inputs)  # shape: [1, 10]
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            total_samples += labels.size(0)

            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        avg_acc = (running_accuracy / total_samples) * 100
        print(f'training loss: {avg_loss}, accuracy: {avg_acc}')

        print()


    def validate_epoch():
        net.train(False)
        running_loss = 0.0
        correct = 0
        total_samples = 0

        for data in validation_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            with torch.no_grad():
                outputs, _ = net(inputs)
                correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total_samples += labels.size(0)

        avg_loss = running_loss / len(validation_loader)
        avg_acc = (correct / total_samples) * 100
        print(f'validation loss: {avg_loss}, validation accuracy: {avg_acc}\n')


    def train(num_epochs=1):
        for epoch_i in range(num_epochs):
            print(f'training epoch {epoch_i}')
            train_epoch()
            validate_epoch()
        print('training complete')


    def test():
        net.eval()
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs, _ = net(inputs)
                correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                total_samples += labels.size(0)

        print(f'test accuracy: {correct / total_samples * 100}')


    train()
    # test()

    bins, edge_vals = encode_bins(50)

    print(train_test_hd(1000, train_loader, test_loader))

# results (tested on 100 batches of size 10):
# 60.8% accuracy, trained w/ 1000 examples
