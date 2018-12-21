'''
Least Squares Fit
'''
import numpy as np
import matplotlib.pyplot as plt


def visualize(X, Y, W, title, labels):
    '''
    Draw graphs
    :param X: list of x
    :param Y: list of y
    :param W: (w0, w1)
    :param title: title
    :param labels: (xlabel, ylabel)
    :return: None
    '''
    axes = plt.gca()
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    axes.scatter(X, Y)
    Y_fit = get_line(W, X)
    axes.plot(X, Y_fit)
    plt.show()


def get_line(W, X):
    '''
    Calculate Y = WX
    :param W: (w0, w1)
    :param X: list of x
    :return: Y
    '''
    X = np.vstack([[1] * len(X), X])
    return W @ X


def linear_regression(X, Y):
    '''
    Exact solution for W = YX_pseudo_inv
    :param X: array of x
    :param Y: array of y
    :return: W
    '''
    X = np.vstack([[1] * len(X), X])
    return Y @ (np.transpose(X) @ np.linalg.inv(X @ np.transpose(X)))


def calc_gradient(X, Y, W):
    '''
    Calculate gradient
    :param X: array of x
    :param Y: array of y
    :param W: (w0, w1)
    :return: gradient
    '''
    gradient = np.empty(2)
    for i in range(0, len(Y)):
        gradient = gradient + (Y[i] - (W[0] + W[1] * X[i])) * np.array([1, X[i]])
    return - 2 * gradient


def gradient_descent(X, Y):
    '''
    Gradient Descent algorithm
    :param X: array of x
    :param Y: array of y
    :return: W
    '''
    W = np.array([0.15, 1])
    eta = 0.001/len(Y)
    epochs = 0
    while True:
        W_prev = W
        gradient = calc_gradient(X, Y, W)
        W = W - eta * gradient
        epochs += 1
        if np.array(abs(W - W_prev) < 10**-4).all():
            break
    print(epochs)
    return W


def main():
    '''
    Main
    :return: None
    '''
    X = np.arange(1, 51, 1)
    Y = np.array([X[i] + np.random.uniform(-1, 1) for i in range(0, len(X))])
    W_lr = linear_regression(X, Y)
    W_gd = gradient_descent(X, Y)
    print(W_lr)
    visualize(X, Y, W_lr, 'Linear Regression', ('x', 'y'))
    visualize(X, Y, W_gd, 'Gradient Descent', ('x', 'y'))


if __name__ == '__main__':
    main()
