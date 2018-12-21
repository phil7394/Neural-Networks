'''
Digit Classifier using Multi category perceptron
'''
import gzip
import random
# from prog_bar import progbar

import numpy as np
import struct
import pylab


def read_idx(filename):
    '''
    Parse idx file and return numpy array

    :param filename: idx file
    :return: numpy array
    '''
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def count_errors(W, n, images, labels):
    '''
    Count number of misclassifications

    :param W: weight vector
    :param n: number of samples
    :param images: data
    :param labels: desired labels
    :return: number of misclassifications
    '''
    error_count = 0
    for i in range(0, n):
        V = W @ np.transpose(np.matrix(images[i].reshape(-1)))
        max_v_index = np.argmax(V)
        if labels[i] != max_v_index:
            error_count += 1
    return error_count


def main():
    test_images = read_idx('t10k-images-idx3-ubyte.gz')
    test_labels = read_idx('t10k-labels-idx1-ubyte.gz')
    train_images = read_idx('train-images-idx3-ubyte.gz')
    train_labels = read_idx('train-labels-idx1-ubyte.gz')
    eta = 1  # learning rate
    n_list = [50, 1000, 60000]  # number of samples
    # n_list = [60000, 60000, 60000]
    epsilon = 0.14  # error threshold
    epoch_limit = 50  # limit on epoch
    for n in n_list:
        W = np.matrix([[random.uniform(-1, 1) for i in range(784)] for j in range(10)])  # initial weights
        epoch = 0
        errors = []
        print('\nn = ', n)
        while True:
            errors.append(count_errors(W, n, train_images, train_labels))
            print('\n epoch =', epoch)
            epoch += 1
            for i in range(0, n):
                desired_label = np.matrix([1 if k == train_labels[i] else 0 for k in range(0, 10)]) # desired label
                predicted_label = np.matrix(
                    [1 if v >= 0 else 0 for v in W @ np.transpose(np.matrix(train_images[i].reshape(-1)))]) # predicted label
                W = W + eta * np.transpose(desired_label - predicted_label) @ np.matrix(train_images[i].reshape(-1)) # weight update
                # progbar(i, n - 1, 20)  # progress bar
            if errors[epoch - 1] / n <= epsilon or epoch == epoch_limit: # check for error threshold
                train_perc_error = errors[epoch - 1] / n
                break

        # plot epochs vs error
        pylab.plot(list(range(0, epoch)), errors)
        pylab.xlabel('epochs')
        pylab.ylabel('errors')
        pylab.xlim(0, epoch)
        pylab.ylim(0, n)
        pylab.title('n = ' + str(n))
        pylab.show()
        test_error_count = count_errors(W, test_images.shape[0], test_images, test_labels)
        test_perc_error = test_error_count / test_images.shape[0]
        print('\nTraining error % = {}'.format(train_perc_error))
        print('Testing error % = {}'.format(test_perc_error))


if __name__ == '__main__':
    main()
