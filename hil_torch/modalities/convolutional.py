# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------- Training the CNN ------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

        return x


def train_epoch():
    net.train(True)
    total_loss = 0.0
    correct = 0.0
    total = 0

    i = 0

    for data in train_loader:
        print(f'training data #{i + 1} out of {len(train_loader)}')
        i += 1

        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)  # shape: [1, 10]
        correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()

        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total += labels.size(0)

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    avg_acc = (correct / total) * 100
    print(f'training loss: {avg_loss}, accuracy: {avg_acc}')

    print()


def validate_epoch():
    net.train(False)
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for data in validation_loader:
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = net(inputs)
            correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            total_loss += criterion(outputs, labels).item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(validation_loader)
    avg_acc = (correct / total_samples) * 100
    print(f'validation loss: {avg_loss}, validation accuracy: {avg_acc}\n')


def train(num_epochs=1):
    for epoch_i in range(num_epochs):
        print(f'training epoch {epoch_i}')
        train_epoch()
        validate_epoch()
    print('training complete')


transform = transforms.ToTensor()
device = 'cuda'

batch_size = 10
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

train_set, validation_set = torch.utils.data.random_split(train_set, [50000, 10000])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

net = ConvoNet()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

train()

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------- Hypervector encoding of CNN output signals  --------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# process: pass an image into the network. receive real valued vector of output signals. normalize with tan-h. bin
# each value to the corresponding hypervector. bind to position symbolic vector. bundle across signals to create vector
# representation for the image.

import vector as vec
import numpy as np
from bases import pos_vecs, output_vecs


def encode_bins(num_bins=32):
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    bin_vectors = [vec.Vector()]
    indexes = np.arange(10000)  # using vectors of length 10000

    for i in range(num_bins):
        bin_bits = bin_vectors[-1].bits.clone()
        to_change = np.random.choice(indexes, size=10000 // num_bins, replace=False)
        for idx in to_change:
            bin_bits[idx] ^= 1
        indexes = np.setdiff1d(indexes, to_change)
        bin_vectors.append(vec.Vector(bin_bits))

    return bin_vectors, bin_edges


def encode_image(image):
    signal_vectors = []
    net.eval()
    with torch.no_grad():
        cnn_output = net(image.unsqueeze(0))  # the CNN expects a batch, give the image a batch index
        binned = np.digitize(torch.tanh(cnn_output).cpu().numpy(), bins=edges)[0]
        for i in range(len(binned)):  # bind the signal hypervector to the component position hypervector
            signal_vectors.append(vec.xor(bins[binned[i] - 1], pos_vecs[i]))  # need -1 to adjust from digitize
    return vec.consensus_sum(signal_vectors)


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- Training and Testing the HIL   --------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def train_test(num_examples=1000):
    net.eval()
    # train
    class_vecs = {i: [] for i in range(10)}
    example_count = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if example_count == num_examples:
                break
            inputs, labels = data[0].to(device), data[1].to(device)
            for image, label in zip(inputs, labels):
                class_vecs[label.item()].append(encode_image(image))
                example_count += 1

    sum_vecs = [vec.consensus_sum(class_vecs[i]) for i in range(10)]
    bound_vecs = [vec.xor(sum_vecs[i], output_vecs[i]) for i in range(10)]
    hil = vec.consensus_sum(bound_vecs)

    # test
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            for image, label in zip(inputs, labels):
                if total == 1000:  # test on 1000 examples
                    break
                img_vec = encode_image(image)
                result = vec.xor(img_vec, hil)
                min_hd = 10000
                prediction = -1
                for j in range(10):
                    cur_hd = vec.hamming_distance(result, output_vecs[j])
                    if cur_hd < min_hd:
                        min_hd = cur_hd
                        prediction = j
                if prediction == label.item():
                    correct += 1
                total += 1

    return correct / 1000.0


bins, edges = encode_bins()

print(train_test(50))

# 50,000 training examples, 71% accuracy
# 50 training examples, 63% accuracy
