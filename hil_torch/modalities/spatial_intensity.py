# hil using pytorch, using a spatial intensity encoding

import tensorflow as tf
import vector as vec
from intensities import intensity_vecs
from bases import row_vecs, col_vecs, output_vecs
import matplotlib.pyplot as plt

def encode(matrix):
    pixels = []
    for row in range(0, len(matrix)):
        for col in range(0, len(matrix[0])):
            pixel_vec = intensity_vecs[matrix[row][col]]
            pixel_vec = vec.xor(pixel_vec, row_vecs[row])
            pixel_vec = vec.xor(pixel_vec, col_vecs[col])
            pixels.append(pixel_vec)
    return vec.consensus_sum(pixels)


# intakes the resultant vector from t XOR hil, returns the predicted output class via minimizing hamming distance
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
        img = encode(train_images[i])
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
        t = encode(test_images[i])
        res = vec.xor(hil, t)

        pre = predict(res)
        expected = test_labels[i]

        if pre == expected:
            correct += 1

    return correct / 100

accuracies = []
training_sizes = []
size = 10

while size < 8000:
    training_sizes.append(size)
    accuracy = train_and_test(size) * 100
    accuracies.append(accuracy)
    size *= 2

# Plotting the results
plt.plot(training_sizes, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Number of training examples')
plt.ylabel('Prediction accuracy (%) ')
plt.title('Effect of Increased Training Examples on Prediction Accuracy')
plt.grid(True)
plt.show()
