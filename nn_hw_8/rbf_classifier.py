import numpy as np
import matplotlib.pyplot as plt


def sun(x):
    return (x[1] - 0.8) ** 2 + (x[0] - 0.5) ** 2 < 0.15 ** 2


def mountain(x):
    return x[1] < 1 / 5 * np.sin(10 * x[0]) + 0.3


def desired_output(x):
    return 1.0 if mountain(x) or sun(x) else -1.0


def init_clf(n):
    X = np.random.uniform(0, 1, (n, 2))
    D = np.array([desired_output(x) for x in X])
    C_plus = [X[i] for i in range(len(X)) if D[i] == 1.0]
    C_minus = [X[i] for i in range(len(X)) if D[i] == -1.0]
    return C_minus, C_plus, D, X


def k_means(C_minus, k, max_iter=100):
    centroids = C_minus[np.random.choice(C_minus.shape[0], k, replace=False)]
    i = 0
    final_clusters = {}
    while i <= max_iter:
        clusters = {}
        for x in C_minus:
            dist_list = [np.linalg.norm(x - c) for c in centroids]
            min_c = dist_list.index(min(dist_list))
            center = tuple(centroids[min_c])
            if clusters.get(center, False):
                clusters[center].append(x)
            else:
                clusters[center] = [x]
        prev_centroids = centroids
        centroids = np.array([np.mean(pts, axis=0) for pts in clusters.values()])
        if (centroids == prev_centroids).all():
            final_clusters = clusters
            break
        i += 1
    betas = []
    for center, pts in final_clusters.items():
        # sigma = np.sum(np.linalg.norm(np.array(pts) - center, axis=1)) / len(pts)
        # betas.append(1 / (2 * sigma ** 2) if sigma != 0 else 1 / 2)
        betas.append(1 / (2 * np.var(pts)))
    return centroids, betas


def plot_centers(c_minus_centers, c_plus_centers, C_minus, C_plus):
    plt.scatter(*zip(*c_minus_centers), c='red', s=100, marker='_', label='C-')
    plt.scatter(*zip(*c_plus_centers), c='green', s=100, marker='+', label='C+')
    plt.scatter(*zip(*C_minus), c='red', s=1, label='C-')
    plt.scatter(*zip(*C_plus), c='green', s=1, label='C+')
    plt.legend(loc='best')
    plt.show()


def rbf(x, c_minus_centers, c_plus_centers, betas):
    centers = np.append(c_minus_centers, c_plus_centers, axis=0)
    X = np.array([x] * centers.shape[0])
    rbf = np.exp(-betas * np.power(np.linalg.norm(X - centers, axis=1), 2))
    return rbf


def run_next_epoch(X, C_minus, C_plus, c_minus_centers, c_plus_centers, W, eta, betas):
    mis_clf = 0
    for s in X:
        x = np.append(1, rbf(s, c_minus_centers, c_plus_centers, betas))
        if x @ W >= 0 and not np.isin(s, C_plus).all():
            W = W - eta * x
            mis_clf = mis_clf + 1
        elif x @ W < 0 and not np.isin(s, C_minus).all():
            W = W + eta * x
            mis_clf = mis_clf + 1
    return W, mis_clf


def get_hyper_planes(c_minus_centers, c_plus_centers, W, betas):
    x = np.linspace(0.0, 1.0, num=1000)
    y = np.linspace(0.0, 1.0, num=1000)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    centers = np.append(c_minus_centers, c_plus_centers, axis=0)
    k = centers.shape[0]
    X = np.repeat(xy, k, axis=0).reshape(-1, k, 2)
    X = np.exp(-betas * np.power(np.linalg.norm(X - centers, axis=2), 2))
    b = np.matrix([1.0] * xy.shape[0])
    X = np.hstack((b.T, X))
    G = X @ W
    G = np.asarray(G).reshape(-1)
    H = xy[(-0.001 < G) & (G < 0.001)]
    return H


def plot_boundary(c_minus_centers, c_plus_centers, C_minus, C_plus, H):
    plt.scatter(*zip(*c_minus_centers), c='red', s=100, marker='_', label='C-')
    plt.scatter(*zip(*c_plus_centers), c='green', s=100, marker='+', label='C+')
    plt.scatter(*zip(*C_minus), c='red', s=1, label='C-')
    plt.scatter(*zip(*C_plus), c='green', s=1, label='C+')
    plt.scatter(*zip(*H), c='blue', s=1, label='H')
    plt.legend(loc='best')
    plt.show()


def main():
    # np.random.seed(100) #1994 -- 1231
    n = 100
    k = 4
    C_minus, C_plus, D, X = init_clf(n)

    c_minus_centers, c_minus_betas = k_means(np.array(C_minus), k // 2)
    c_plus_centers, c_plus_betas = k_means(np.array(C_plus), k // 2)
    betas = np.array(c_minus_betas + c_plus_betas)
    plot_centers(c_minus_centers, c_plus_centers, C_minus, C_plus)
    W = np.random.uniform(-1, 1, k + 1) # -1,1 -- -1.0,1.0
    eta = 0.001 # 0.01 -- 0.001
    epoch = 0
    mis_clf = n
    while mis_clf > 12 and epoch < 1000: #0--12
        # prev_misclf = mis_clf
        W, mis_clf = run_next_epoch(X, C_minus, C_plus, c_minus_centers, c_plus_centers, W, eta, betas)
        # if mis_clf > prev_misclf:
        #     eta = 0.9 * eta
        epoch = epoch + 1
        print(epoch, mis_clf, eta)
    print('W_final={}'.format(W))
    H = get_hyper_planes(c_minus_centers, c_plus_centers, W, betas)
    plot_boundary(c_minus_centers, c_plus_centers, C_minus, C_plus, H)


if __name__ == '__main__':
    main()
