# hil to test the model's memoization of example vectors

# fix so... trained on random vectors instead of semantic (encoded ones)
# and then, (not sure but maybe) test on encoded vectors

import tensorflow as tf
import vector as vec
from intensities import intensity_vecs
from bases import row_vecs, col_vecs, output_vecs
import matplotlib.pyplot as plt


def predict(vector):
    min_hd = 10000
    prediction = -1
    for i in range(10):
        cur_hd = vec.hamming_distance(vector, output_vecs[i])
        if cur_hd < min_hd:
            min_hd = cur_hd
            prediction = i
    return prediction


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


def train_and_test(num_examples):
    # training
    class_vecs = {i: [] for i in range(10)}

    for i in range(num_examples):  # append each encoded image to the corresponding class list
        print(f'cur i: {i}')
        img = vec.Vector()  # switch from sematic to lookup vector
        class_vecs[train_labels[i]].append(img)

    sums = []
    for i in range(10):  # create a summed vector representation for each class
        cons = vec.consensus_sum(class_vecs[i])
        if cons is not None:
            sums.append(cons)

    bound_vecs = []
    count = 0
    for sum_vec in sums:  # bind each summed representation
        bound_vecs.append(vec.xor(sum_vec, output_vecs[count]))
        count += 1

    hil = vec.consensus_sum(bound_vecs)  # consensus sum the bound representations into the hil

    # testing
    correct = 0
    for i in range(100):
        t = vec.Vector()  # switch from sematic to lookup vector
        res = vec.xor(hil, t)

        pre = predict(res)
        expected = test_labels[i]

        if pre == expected:
            correct += 1

    return correct / 100


accuracies = []
training_sizes = []
size = 10

while size < 50000:
    training_sizes.append(size)
    accuracy = train_and_test(size) * 100
    accuracies.append(accuracy)
    size *= 2

# Plotting the results
plt.plot(training_sizes, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Number of training examples')
plt.ylabel('Prediction accuracy (%) ')
plt.title('Effect of Increased Training Examples on Prediction Accuracy using Lookup Vectors')
plt.grid(True)
plt.show()