import gzip
import struct
import math
import sys
import matplotlib.pyplot as plt
import numpy as np


def read_idx(filename):
    """
    Parse idx file and return numpy array

    :param filename: idx file
    :return: numpy array
    """
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def init_network(input_size, output_size):
    neu_net = []
    # w_0 = [np.random.uniform(low=-5, high=5, size=input_size) for _ in range(0, output_size)]
    # b_0 = np.random.uniform(low=-1, high=1, size=output_size)
    r_w = np.sqrt(6 / (input_size + output_size))
    r_b = np.sqrt(6 / (output_size + output_size))
    w_0 = [np.random.uniform(0, r_w, input_size) for _ in range(0, output_size)]
    b_0 = np.random.uniform(0, r_b, output_size)
    w = [list(np.hstack((w_0[i], b_0[i]))) for i in range(0, len(b_0))]
    output_layer = [{'weights': w[i]} for i in range(len(w))]
    neu_net.append(output_layer)
    return neu_net


# def init_network(input_size, hidden_size, output_size):
#     neu_net = []
#
#     # w_0 = [np.random.uniform(low=-5, high=5, size=input_size) for _ in range(hidden_size)]
#     # b_0 = np.random.uniform(low=-1, high=1, size=hidden_size)
#     # w_1 = [np.random.uniform(low=-5, high=5, size=hidden_size) for _ in range(output_size)]
#     # b_1 = np.random.uniform(low=-1, high=1, size=output_size)
#
#     r_w_0 = np.sqrt(6 / (input_size + hidden_size))
#     r_b_0 = np.sqrt(6 / (hidden_size + output_size))
#     r_w_1 = np.sqrt(6 / (hidden_size + output_size))
#     r_b_1 = np.sqrt(6 / (output_size + output_size))
#     w_0 = [np.random.uniform(low=0, high=r_w_0, size=input_size) for _ in range(hidden_size)]
#     b_0 = np.random.uniform(low=-1, high=r_b_0, size=hidden_size)
#     w_1 = [np.random.uniform(low=-5, high=r_w_1, size=hidden_size) for _ in range(output_size)]
#     b_1 = np.random.uniform(low=-1, high=r_b_1, size=output_size)
#
#     l1_w = [list(np.hstack((w_0[i], b_0[i]))) for i in range(hidden_size)]
#     l2_w = [list(np.hstack((w_1[i], b_1[i]))) for i in range(output_size)]
#     hidden_layer = [{'weights': l1_w[i]} for i in range(len(l1_w))]
#     neu_net.append(hidden_layer)
#     output_layer = [{'weights': l2_w[i]} for i in range(len(l2_w))]
#     neu_net.append(output_layer)
#     return neu_net


def calc_v(weights, inputs):
    v = weights[-1]
    for i in range(len(weights) - 1):
        v += weights[i] * inputs[i]
    return v


def activation_func(v):
    return math.tanh(v)


def activation_func_deriv(v):
    return 1 - (math.tanh(v) ** 2)


def filter_output(outputs):  # TODO np.array
    max_index = np.array(outputs).argmax()
    filtered_output = [0] * len(outputs)
    filtered_output[max_index] = 1
    return filtered_output


def forward_propagate(net, init_inputs):
    inputs = init_inputs  # TODO np.array
    for layer in range(0, len(net)):
        outputs = []  # TODO np.array
        for neuron in net[layer]:
            neuron['v'] = calc_v(neuron['weights'], inputs)
            neuron['output'] = activation_func(neuron['v'])
            outputs.append(neuron['output'])
        inputs = outputs
    return filter_output(inputs)


def backward_propagate(net, desired):
    for i in reversed(range(len(net))):
        layer = net[i]
        errors = list()
        if i != len(net) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in net[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(desired[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * activation_func_deriv(neuron['v'])


def update_weights(net, x, eta):
    for layer in range(len(net)):
        inputs = x
        if layer != 0:
            inputs = [neuron['output'] for neuron in net[layer - 1]]
        for neuron in net[layer]:
            for j in range(len(inputs)):
                neuron['weights'][j] += eta * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += eta * neuron['delta'] * 1


def calc_mse(y_pred, y_act):
    mse = 0
    for i in range(0, len(y_act)):
        mse += np.linalg.norm(np.array(y_act[i]) - np.array(y_pred[i]), ord=2) ** 2
    return mse / len(y_act)


def progbar(curr, total, full_progbar):
    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    print('\r', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()


def train_network(net, train_data, test_data, eta, mse_threshold):
    epoch = 0
    prev_mse = 0
    tr_mse_list = []
    te_mse_list = []
    tr_misclf_list = []
    te_misclf_list = []

    while True:
        print('Training epoch {}.....'.format(epoch + 1))
        for i in range(0, len(train_data['x'])):
            forward_propagate(net, train_data['x'][i])
            backward_propagate(net, train_data['y'][i])
            update_weights(net, train_data['x'][i], eta)
            progbar(i + 1, len(train_data['x']), 10)
        print()
        epoch += 1
        print('Testing on train data.....')
        tr_y_pred = test_network(net, train_data, decode=False)
        print('Testing on test data.....')
        te_y_pred = test_network(net, test_data, decode=False)
        tr_misclf_list.append(count_misclfs(tr_y_pred, train_data['y']))
        te_misclf_list.append(count_misclfs(te_y_pred, test_data['y']))
        tr_mse_list.append(calc_mse(tr_y_pred, train_data['y']))
        te_mse_list.append(calc_mse(te_y_pred, test_data['y']))
        te_accuracy = (1 - te_misclf_list[-1] / len(test_data['y'])) * 100
        tr_accuracy = (1 - tr_misclf_list[-1] / len(train_data['y'])) * 100
        print('>epoch={}, eta={:.7f}, tr_mse={:.7f}, te_mse={:.7f}, tr_accuracy={:.4f}%, te_accuracy={:.4f}%'.format(
            epoch, eta,
            tr_mse_list[-1],
            te_mse_list[-1],
            tr_accuracy,
            te_accuracy))
        if te_accuracy >= 95 or tr_mse_list[-1] <= mse_threshold or eta < 10 ** -7:
            break
        if tr_mse_list[-1] >= prev_mse and epoch > 1:
            eta = 0.9 * eta
        prev_mse = tr_mse_list[-1]
    return tr_mse_list, te_mse_list, tr_misclf_list, te_misclf_list, te_accuracy


def count_misclfs(y_pred, y_act):
    count = 0
    for i in range(0, len(y_act)):
        if y_pred[i] != y_act[i]:
            count += 1
    return count


def one_hot_decode(arr):
    return np.array(arr).argmax()


def predict(net, x, decode=True):
    return one_hot_decode(forward_propagate(net, x)) if decode else forward_propagate(net, x)


def one_hot_encode(value, size):
    encoded_val = [0] * size
    if value < size:
        encoded_val[value] = 1
    return encoded_val


def normalize(arr):
    return arr / np.linalg.norm(arr, ord=2)


def preprocess_data(images, labels, n=None):
    data = {}
    if not n:
        n = len(images)
    data['x'] = [normalize(images[i].reshape(-1)) for i in range(0, n)]
    data['y'] = [one_hot_encode(labels[i], size=10) for i in range(0, n)]
    return data


def test_network(net, data, decode=True):
    y_pred = []
    for i in range(0, len(data['x'])):
        y_pred.append(predict(net, data['x'][i], decode=decode))
        progbar(i + 1, len(data['x']), 20)
    print()
    return y_pred


def calc_accuracy(y_pred, y_act):
    return np.mean(np.array(y_pred) == np.array(y_act)) * 100


def plot_data(tr_mse_list, te_mse_list, tr_misclf_list, te_misclf_list):
    plt.plot(tr_mse_list)
    plt.title('Training: Epochs vs Energy')
    plt.xlabel('epochs')
    plt.ylabel('energy')
    plt.show()
    plt.plot(te_mse_list)
    plt.title('Testing: Epochs vs Energy')
    plt.xlabel('epochs')
    plt.ylabel('energy')
    plt.show()
    plt.plot(tr_misclf_list)
    plt.title('Training: Epochs vs Misclassifications')
    plt.xlabel('epochs')
    plt.ylabel('misclassifications')
    plt.show()
    plt.plot(te_misclf_list)
    plt.title('Testing: Epochs vs Misclassifications')
    plt.xlabel('epochs')
    plt.ylabel('misclassifications')
    plt.show()


def main():
    test_images = read_idx('t10k-images-idx3-ubyte.gz')
    test_labels = read_idx('t10k-labels-idx1-ubyte.gz')
    train_images = read_idx('train-images-idx3-ubyte.gz')
    train_labels = read_idx('train-labels-idx1-ubyte.gz')
    train_data = preprocess_data(train_images, train_labels, n=60000)
    test_data = preprocess_data(test_images, test_labels)
    net = init_network(input_size=784, output_size=10)
    # net = init_network(input_size=784, hidden_size=10, output_size=10)
    tr_mse_list, te_mse_list, tr_misclf_list, te_misclf_list, te_accuracy = train_network(net, train_data, test_data,
                                                                                          eta=0.5,
                                                                                          mse_threshold=0.01)
    # print('Final testing on test data.....')
    print('\nTest Accuracy {:.2f}%'.format(te_accuracy))
    plot_data(tr_mse_list, te_mse_list, tr_misclf_list, te_misclf_list)


if __name__ == '__main__':
    main()
