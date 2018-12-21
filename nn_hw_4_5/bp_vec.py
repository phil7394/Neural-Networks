import sys

import numpy as np

import os
import gzip
import matplotlib.pyplot as plt

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'


# Download and import the MNIST dataset from Yann LeCun's website.
# Reserve 10,000 examples from the training set for validation.
# Each image is an array of 784 (28x28) float values  from 0 (white) to 1 (black).
def load_data(one_hot=True, reshape=None, validation_size=10000):
    x_tr = load_images('train-images-idx3-ubyte.gz')
    y_tr = load_labels('train-labels-idx1-ubyte.gz')
    x_te = load_images('t10k-images-idx3-ubyte.gz')
    y_te = load_labels('t10k-labels-idx1-ubyte.gz')

    x_tr = x_tr[:-validation_size]
    y_tr = y_tr[:-validation_size]

    if one_hot:
        y_tr, y_te = [to_one_hot(y) for y in (y_tr, y_te)]

    if reshape:
        x_tr, x_te = [x.reshape(*reshape) for x in (x_tr, x_te)]

    return x_tr, y_tr, x_te, y_te


def load_images(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / np.float32(256)


def load_labels(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# Download the file, unless it's already here.
def maybe_download(filename):
    if not os.path.exists(filename):
        from urllib import urlretrieve
        print("Downloading %s" % filename)
        urlretrieve(DATA_URL + filename, filename)


# Convert class labels from scalars to one-hot vectors.
def to_one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w), 0))
    return a


def grads(X, Y, weights):
    grads = np.empty_like(weights)
    a = feed_forward(X, weights)
    delta = a[-1] - Y
    grads[-1] = a[-2].T.dot(delta)
    for i in range(len(a) - 2, 0, -1):
        delta = (a[i] > 0) * delta.dot(weights[i].T)
        grads[i - 1] = a[i - 1].T.dot(delta)
    return grads / len(X)


def count_misclfs(y_pred, y_act):
    return np.count_nonzero(y_pred != y_act)
    # count = 0
    # for i in range(0, len(y_act)):
    #     if y_pred[i] != y_act[i]:
    #         count += 1
    # return count


def calc_mse(y_pred, y_act):
    mse = 0
    for i in range(0, len(y_act)):
        mse += np.linalg.norm(np.array(y_act[i]) - np.array(y_pred[i]), ord=2) ** 2
    return mse / len(y_act)


def plot_data(tr_mse_list, te_mse_list, tr_misclf_list, te_misclf_list):
    plt.plot(tr_mse_list, label='Training')
    plt.plot(te_mse_list, label='Testing')
    plt.gca().legend(loc='best')
    plt.title('Epochs vs Energy')
    plt.xlabel('epochs')
    plt.ylabel('energy')
    # plt.show()
    # plt.title('Testing: Epochs vs Energy')
    # plt.xlabel('epochs')
    # plt.ylabel('energy')
    plt.show()
    plt.plot(tr_misclf_list, label='Training')
    plt.plot(te_misclf_list, label='Testing')
    plt.gca().legend(loc='best')
    plt.title('Epochs vs Misclassifications')
    plt.xlabel('epochs')
    plt.ylabel('misclassifications')
    plt.show()

#
# def progbar(curr, total, full_progbar):
#     frac = curr / total
#     filled_progbar = round(frac * full_progbar)
#     print('\r', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='')
#     sys.stdout.flush()


trX, trY, teX, teY = load_data()
weights = [np.random.randn(*w) * 0.1 for w in [(784, 100), (100, 10)]]
num_epochs, batch_size, learn_rate = 30, 20, 0.1

tr_misclf_list = []
te_misclf_list = []
tr_mse_list = []
te_mse_list = []
epoch = 0
while True:
    for j in range(0, len(trX), batch_size):
        X, Y = trX[j:j + batch_size], trY[j:j + batch_size]
        weights -= learn_rate * grads(X, Y, weights)
        # progbar(j, len(trX), 30)
    epoch += 1
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    print('>', epoch, np.mean(prediction == np.argmax(teY, axis=1)))

    tr_pred = feed_forward(trX, weights)[-1]
    te_pred = feed_forward(teX, weights)[-1]
    tr_misclf_list.append(count_misclfs(np.argmax(tr_pred, axis=1), np.argmax(trY, axis=1)))
    te_misclf_list.append(count_misclfs(np.argmax(te_pred, axis=1), np.argmax(teY, axis=1)))
    tr_mse_list.append(calc_mse(tr_pred, trY))
    te_mse_list.append(calc_mse(te_pred, teY))
    te_accuracy = (1 - te_misclf_list[-1] / 10000) * 100
    if te_accuracy >= 95:
        break

plot_data(tr_mse_list, te_mse_list, tr_misclf_list, te_misclf_list)
