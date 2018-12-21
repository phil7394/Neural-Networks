"""
Perceptron Training Algorithm
"""
import random
import numpy as np
import matplotlib.pyplot as plt


def evaluate(w, x):
    """
    Evaluates equation for line

    :param w: list of weights
    :param x: x coordinate
    :return: y coordinate
    """
    return -(w[0] / w[2]) - (w[1] / w[2]) * x


def plot_epoch_mis_clfs(epoch, mis_clfs, title):
    """
    Plot epoch vs misclassifications

    :param epoch: epoch
    :param mis_clfs: misclassifications list
    :param title: title for plot
    :return: None
    """
    plt.plot(list(range(1, epoch + 1)), mis_clfs)
    plt.gca().set_xlabel('epochs')
    plt.gca().set_ylabel('misclassifications')
    plt.gca().set_title(title)
    plt.show()


def run_next_epoch(S, S_0, S_1, W_prime, eta):
    """
    Run perceptron training algorithm for an epoch

    :param S: set of all training data
    :param S_0: class 0 points
    :param S_1: class 1 points
    :param W_prime: previous weights
    :param eta: learning rate
    :return: new weights, no. of misclassifications
    """
    mis_clf = 0
    for s in S:
        x = np.array([1, s[0], s[1]])
        if x @ W_prime >= 0 and s not in S_1:
            W_prime = W_prime - eta * x
            mis_clf = mis_clf + 1
        elif x @ W_prime < 0 and s not in S_0:
            W_prime = W_prime + eta * x
            mis_clf = mis_clf + 1
    return W_prime, mis_clf


def plot_perceptron(S_0_x, S_0_y, S_1_x, S_1_y, X, Y, title):
    """
    Plot a perceptron given points for line, class 0 and class 1

    :param S_0_x: x points of class 0
    :param S_0_y: y points of class 0
    :param S_1_x: x points of class 1
    :param S_1_y: x points of class 1
    :param X: x points of line
    :param Y: y points of line
    :param title: title for the plot
    :return: None
    """
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_xlabel('x1')
    axes.set_ylim([-1, 1])
    axes.set_ylabel('x2')
    axes.set_title(title)
    line_plt, = plt.plot(X, Y)
    S_0_plt = plt.scatter(S_0_x, S_0_y, s=50, marker='_', linewidths=2)
    S_1_plt = plt.scatter(S_1_x, S_1_y, s=50, marker='+', linewidths=2)
    plt.legend((line_plt, S_0_plt, S_1_plt), ('boundary', 'S0', 'S1'), loc=1)
    plt.show()


def main():
    W = np.array([random.uniform(-0.25, 0.25),
                  random.uniform(-1.0, 1.0),
                  random.uniform(-1.0, 1.0)]) # optimal weights [w0 w1 w2]
    print('W={}'.format(W))
    X = np.array([-1, 1]) # x points for line
    Y = list(map(lambda x: evaluate(W, x), X)) # solve for y
    n_list = [100, 1000] # list of training size
    S = [] # set of all training data
    S_0 = [] # set of training data in class 0
    S_1 = [] # set of training data in class 1

    for n in n_list:
        for i in range(1, n + 1):
            S.append((random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))) # uniformly and randomly sample training set S
        for s in S:
            x = np.array([1, s[0], s[1]])
            S_1.append(s) if x @ W >= 0 else S_0.append(s) # assign classes 0 and 1

        # extract x and y points from each class for plotting
        S_0_x = [s[0] for s in S_0]
        S_0_y = [s[1] for s in S_0]
        S_1_x = [s[0] for s in S_1]
        S_1_y = [s[1] for s in S_1]

        plot_perceptron(S_0_x, S_0_y, S_1_x, S_1_y, X, Y, title='n=' + str(n)) # plot the line, class 0 and class 1 points

        W_prime = np.array([random.uniform(-1.0, 1.0),
                            random.uniform(-1.0, 1.0),
                            random.uniform(-1.0, 1.0)]) # [w0' w1' w2']
        print('n={}, W_prime={}'.format(n, W_prime))
        eta_list = [1, 10, 0.1] # list of different learning rates
        for eta in eta_list:
            epoch = 1
            W_prime_2, mis_clf = run_next_epoch(S, S_0, S_1, W_prime, eta) # run training algorithm for next epoch
            mis_clfs = [mis_clf] # list of number of misclassifications

            while mis_clf > 0:
                W_prime_2, mis_clf = run_next_epoch(S, S_0, S_1, W_prime_2, eta)
                mis_clfs.append(mis_clf)
                epoch = epoch + 1
            print('n={}, W_final={}'.format(n, W_prime_2))
            X = np.array([-1, 1]) # x points for plotting line with trained weights
            Y = list(map(lambda x: evaluate(W_prime_2, x), X)) # solve for y
            plot_perceptron(S_0_x, S_0_y, S_1_x, S_1_y, X, Y, title='n=' + str(n) + ', eta=' + str(eta)) # plot trained line, class 0 and class 1

            plot_epoch_mis_clfs(epoch, mis_clfs, title='n=' + str(n) + ', eta=' + str(eta)) # plot epoch vs misclassifications


if __name__ == '__main__':
    main()
