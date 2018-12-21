import math
import random

import matplotlib.pyplot as plt
import numpy as np


def init_network(input_size, hidden_size, output_size):
    neu_net = []
    w_0 = np.random.uniform(low=-5, high=5, size=input_size * hidden_size)
    b_0 = np.random.uniform(low=-1, high=1, size=hidden_size)
    w_1 = np.random.uniform(low=-5, high=5, size=hidden_size * output_size)
    b_1 = np.random.uniform(low=-1, high=1, size=output_size)
    w = list(zip(w_0, b_0))
    hidden_layer = [{'weights': list(w[i])} for i in range(len(w))]
    neu_net.append(hidden_layer)
    output_layer = [{'weights': list(np.hstack((w_1, b_1)))}]
    neu_net.append(output_layer)
    return neu_net


def calc_v(weights, inputs):
    v = weights[-1]
    for i in range(len(weights) - 1):
        v += weights[i] * inputs[i]
    return v


def activation_func(v, layer):
    return v if layer == 1 else math.tanh(v)


def activation_func_deriv(v, layer):
    return 1 if layer == 1 else 1 - (math.tanh(v) ** 2)


def forward_propagate(net, init_inputs):
    inputs = init_inputs
    for layer in range(0, len(net)):
        outputs = []
        for neuron in net[layer]:
            neuron['v'] = calc_v(neuron['weights'], inputs)
            neuron['output'] = activation_func(neuron['v'], layer)
            outputs.append(neuron['output'])
        inputs = outputs
    return inputs


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
                errors.append(desired - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * activation_func_deriv(neuron['v'], i)


def update_weights(net, x, eta):
    for layer in range(len(net)):
        inputs = x
        if layer != 0:
            inputs = [neuron['output'] for neuron in net[layer - 1]]
        for neuron in net[layer]:
            for j in range(len(inputs)):
                neuron['weights'][j] += eta * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += eta * neuron['delta'] * 1


def train_network(net, train_data, eta, mse_threshold):
    epoch = 0
    prev_mse = 0
    mse_list = []
    while True:
        mse = 0
        for i in range(0, len(train_data['x'])):
            x_train = train_data['x'][i]
            outputs = forward_propagate(net, x_train)
            y_train = train_data['y'][i]
            mse += (y_train[0] - outputs[0]) ** 2
            backward_propagate(net, y_train[0])
            update_weights(net, x_train, eta)
        epoch += 1
        mse = mse / len(train_data['x'])
        print('epoch=%d, eta=%.7f, mse=%.7f' % (epoch, eta, mse))
        if mse <= mse_threshold:
            break
        if mse >= prev_mse and epoch > 1:
            eta = 0.9 * eta
        if eta < 10 ** -7:
            break
        prev_mse = mse
        mse_list.append(mse)
    return mse_list


def predict(net, x):
    return forward_propagate(net, x)


def gen_train_data():
    train_data = {}
    train_data['x'] = [[random.uniform(0, 1)] for i in range(0, 301)]
    v = [random.uniform(-0.1, 0.1) for i in range(0, 301)]
    train_data['y'] = [[math.sin(20 * train_data['x'][i][0]) + 3 * train_data['x'][i][0] + v[i]] for i in range(0, 301)]
    return train_data


def plot_data(train_data, mse_list, y_pred):
    plt.scatter(train_data['x'], train_data['y'], c='r', label='actual', marker='.')
    plt.scatter(train_data['x'], y_pred, c='b', label='predicted', marker='+')
    plt.gca().legend(loc='lower right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Best fit curve')
    plt.show()
    plt.plot(mse_list)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title('MSE vs Epochs')
    plt.show()


def main():
    eta = 12 / 300
    train_data = gen_train_data()
    net = init_network(1, 24, 1)
    mse_list = train_network(net, train_data, eta, mse_threshold=0.01)
    y_pred = [predict(net, x) for x in train_data['x']]
    plot_data(train_data, mse_list, y_pred)


if __name__ == '__main__':
    main()
