import cvxopt
import numpy as np
import matplotlib.pyplot as plt


def sun(x):
    return (x[1] - 0.8) ** 2 + (x[0] - 0.5) ** 2 < 0.15 ** 2


def mountain(x):
    return x[1] < 1 / 5 * np.sin(10 * x[0]) + 0.3


def desired_output(x):
    return 1.0 if mountain(x) or sun(x) else -1.0


def poly_kernel(X, Y, p=5):
    return np.power(np.inner(X, Y) + 1, p)


def init_svm(n):
    X = np.random.uniform(0, 1, (n, 2))
    D = np.array([desired_output(x) for x in X])
    C_plus = [X[i] for i in range(len(X)) if D[i] == 1.0]
    C_minus = [X[i] for i in range(len(X)) if D[i] == -1.0]
    K = poly_kernel(X, X)
    return C_minus, C_plus, D, K, X


def solve_svm_quad_opt(D, K, n):
    P = cvxopt.matrix(np.outer(D, D) * K)
    q = cvxopt.matrix([-1.0] * n)
    G = cvxopt.matrix(np.diag([-1.0] * n))
    h = cvxopt.matrix([0.0] * n)
    A = cvxopt.matrix(D, (1, n))
    b = cvxopt.matrix(0.0)
    opt_sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    return opt_sol


def get_sup_vecs(D, X, alpha, n):
    sup_vec_x = [X[i] for i in range(n) if alpha[i] > 1e-5]
    sup_vec_y = [D[i] for i in range(n) if alpha[i] > 1e-5]
    return sup_vec_x, sup_vec_y


def calc_bias(D, X, alpha, n, sup_vec_x, sup_vec_y):
    bias = sup_vec_y[0] - np.sum(alpha.reshape(n, 1) * D.reshape(n, 1) * poly_kernel(X, sup_vec_x[0].reshape(1, 2)),
                                 axis=0)
    return bias


def get_hyper_planes(D, X, alpha, n, theta):
    x = np.linspace(0.0, 1.0, num=1000)
    y = np.linspace(0.0, 1.0, num=1000)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    K = poly_kernel(X, xy)
    G = np.sum(alpha.reshape(n, 1) * D.reshape(n, 1) * K, axis=0) + theta
    H = xy[(-0.1 < G) & (G < 0.1)]
    H_minus = xy[(-1.1 < G) & (G < -0.9)]
    H_plus = xy[(0.9 < G) & (G < 1.1)]
    return H, H_minus, H_plus


def plot_svm(C_minus, C_plus, H, H_minus, H_plus, sup_vec_x):
    plt.subplots(figsize=(10, 10))
    plt.scatter(*zip(*C_plus), c='red', s=100, marker='+', label='C+')
    plt.scatter(*zip(*C_minus), c='green', s=100, marker='_', label='C-')
    plt.scatter(*zip(*H_plus), c='red', s=1, label='H+')
    plt.scatter(*zip(*H), c='blue', s=1, label='H')
    plt.scatter(*zip(*H_minus), c='green', s=1, label='H-')
    plt.scatter(*zip(*sup_vec_x), s=140, linewidths=2, facecolors='none', marker='o', edgecolors='black',
                label='Support Vectors')
    plt.legend(loc='best')
    plt.show()


def main():
    n = 100
    C_minus, C_plus, D, K, X = init_svm(n)
    opt_sol = solve_svm_quad_opt(D, K, n)
    alpha = np.ravel(opt_sol['x'])
    sup_vec_x, sup_vec_y = get_sup_vecs(D, X, alpha, n)
    theta = calc_bias(D, X, alpha, n, sup_vec_x, sup_vec_y)
    H, H_minus, H_plus = get_hyper_planes(D, X, alpha, n, theta)
    plot_svm(C_minus, C_plus, H, H_minus, H_plus, sup_vec_x)


if __name__ == '__main__':
    main()
