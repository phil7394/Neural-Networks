'''
Gradient Descent vs Newton's Method
f(x, y) = − log(1 − x − y) − log x − log y
'''
import numpy as np
import matplotlib.pyplot as plt


def calc_energy(W):
    '''
    Evaluates for f(x, y) = − log(1 − x − y) − log x − log y
    :param W: (x, y)
    :return: float f(x, y)
    '''
    return - np.math.log(1 - W[0] - W[1]) - np.math.log(W[0]) - np.math.log(W[1])


def calc_gradient(W):
    '''
    Evaluate for gradient of f(x, y)
    :param W: (x, y)
    :return: gradient(f(x, y))
    '''
    return np.array([(1 / (1 - W[0] - W[1])) - (1 / W[0]), (1 / (1 - W[0] - W[1])) - (1 / W[1])])


def calc_hessian_inv(W):
    '''
    Evaluate for Hessian inverse of f(x, y)
    :param W: (x, y)
    :return: H_inv(f(x, y))
    '''
    return np.linalg.inv(np.array([[(1 / (1 - W[0] - W[1]) ** 2) + (1 / W[0] ** 2), (1 / (1 - W[0] - W[1]) ** 2)],
                                   [(1 / (1 - W[0] - W[1]) ** 2), (1 / (1 - W[0] - W[1]) ** 2) + (1 / W[1] ** 2)]]))


def gradient_descent(W, eta):
    '''
    Gradient Descent algorithm
    :param W: (x, y)
    :param eta: learning rate
    :return: (list of weights, list of energy values)
    '''
    W_list = [W]
    energy_list = [calc_energy(W)]
    while True:
        gradient = calc_gradient(W)
        W = W - eta * gradient
        if np.array(abs(W - W_list[-1]) < 10 ** -4).all():
            break
        if not ((W > 0).all() and W.sum() < 1):
            print('Out of domain! Change initial weights', W)
            break
        W_list.append(W)
        energy_list.append(calc_energy(W))
    return W_list, energy_list


def newtons_method(W, eta):
    '''
    Newton's Method algorithm
    :param W: (x, y)
    :param eta: learning rate
    :return: (list of weights, list of energy values)
    '''
    W_list = [W]
    energy_list = [calc_energy(W)]
    while True:
        gradient = calc_gradient(W)
        hessian_inv = calc_hessian_inv(W)
        W = W - eta * hessian_inv @ gradient
        if np.array(abs(W - W_list[-1]) < 10 ** -4).all():
            break
        if not ((W > 0).all() and W.sum() < 1):
            print('Out of domain! Change initial weights', W)
            break
        W_list.append(W)
        energy_list.append(calc_energy(W))
    return W_list, energy_list


def visualize(w_list, nrg_list, w_labels, nrg_labels):
    '''
    Draw graphs for x vs y and energy vs iterations
    :param w_list: list of weights
    :param nrg_list: list of energy values
    :param w_labels: (title, xlabel, ylabel)
    :param nrg_labels: (title, xlabel, ylabel)
    :return: None
    '''
    axes = plt.gca()
    w_x_grid = np.linspace(0, 1, 51)
    w_y_grid = np.linspace(0, 1, 51)
    f_grid = []
    for m in w_x_grid:
        for n in w_y_grid:
            f_grid.append(float('Inf')) if not (m > 0 and n > 0 and m + n < 1) else f_grid.append(calc_energy((m, n)))
    f_grid = np.array(f_grid).reshape(len(w_x_grid), len(w_y_grid))
    X, Y = np.meshgrid(w_x_grid, w_y_grid)
    contours = axes.contour(X, Y, f_grid, 20)
    axes.clabel(contours)
    axes.scatter(*zip(*w_list), s=30, lw=0)

    for j in range(1, len(w_list)):
        axes.annotate('', xy=w_list[j], xytext=w_list[j - 1],
                      arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 2},
                      va='center', ha='center')
    plt.title(w_labels[0])
    plt.xlabel(w_labels[1])
    plt.ylabel(w_labels[2])
    plt.show()

    axes = plt.gca()
    axes.plot(range(0, len(nrg_list)), nrg_list)
    plt.title(nrg_labels[0])
    plt.xlabel(nrg_labels[1])
    plt.ylabel(nrg_labels[2])
    plt.show()


def main():
    '''
    Main
    :return: none
    '''
    W_init = np.array([0.8, 0.05]) # 0.1, 0.35
    eta_gd = 0.01
    eta_nm = 1
    gd_w_list, gd_nrg_list = gradient_descent(W_init, eta_gd)
    nm_w_list, nm_nrg_list = newtons_method(W_init, eta_nm)

    gd_w_labels = ('Gradient Descent', 'x', 'y')
    gd_nrg_labels = ('Gradient Descent', 'iterations', 'f(x,y)')
    nm_w_labels = ('Newton\'s Method', 'x', 'y')
    nm_nrg_labels = ('Newton\'s Method', 'iterations', 'f(x,y)')

    visualize(gd_w_list, gd_nrg_list, gd_w_labels, gd_nrg_labels)
    visualize(nm_w_list, nm_nrg_list, nm_w_labels, nm_nrg_labels)


if __name__ == '__main__':
    main()
