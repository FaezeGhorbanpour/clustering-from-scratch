from scipy import random, linalg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal as mvn


def kmeans(k, iteration):
    min_error = math.inf
    min_index = -1
    whole_center = list()
    for i in range(iteration):
        centers_index = np.random.randint(college_.shape[0], size=k)
        centers = college_[centers_index]
        new_centers = np.zeros([k, college_.shape[1]])

        distance_cluster = np.zeros((college_.shape[0], k))
        while True:
            for c in range(k):
                center = centers[c]
                distance_cluster[:, c] = np.linalg.norm(np.array(college_ - center, dtype=np.float64), axis=1)
            clusters = np.argmin(distance_cluster, axis=1)
            for c in range(k):
                new_centers[c] = np.mean(college_[clusters == c], 0)
            if (centers != new_centers).all():
                break
            centers = np.copy(new_centers)
        whole_center.append(centers)

        # find best iteration
        sum_error = 0
        for c in range(k):
            center = centers[c]
            sum_error += np.sum(np.linalg.norm(np.array(college_[clusters == c] - center, dtype=np.float64), axis=1))
        if sum_error <= min_error:
            min_error = sum_error
            min_index = i

    result_centers = whole_center[min_index]
    for c in range(k):
        center = result_centers[c]
        distance_cluster[:, c] = np.linalg.norm(np.array(college_ - center, dtype=np.float64), axis=1)
    clusters = np.argmin(distance_cluster, axis=1)
    return result_centers, clusters, min_error


college = pd.read_csv('College.csv').values
college_ = college[:, 2:]

k = 2
max_iteration = 100
iter_ = 0
n, d = college_.shape
ll_new = 0.0
ms, clusters, error = kmeans(2, 5)
sigmas = list()
for j in range(k):
    s = np.dot((college_[clusters == j] - ms[j]).T, college_[clusters == j] - ms[j])
    sigmas.append(s)

ps = np.random.rand(k)
ll_old = 0
while True:
    print(iter_)
    if iter_ > max_iteration:
        break

    norms = np.zeros((k, n))
    for i in range(n):
        for j in range(k):
            norms[j, i] = mvn(ms[j], sigmas[j]).pdf(college_[i])

    # E step
    ws = np.zeros((k, n))
    for i in range(n):
        s = np.dot(ps, norms[:, i])
        for j in range(k):
            ws[j, i] = ps[j] * norms[j, i] / s

    # M step
    for j in range(k):
        row = ws[j]
        s = np.dot(row, np.ones(n))

        ms[j] = np.dot(row, college_) / s

        dif = college_ - np.tile(ms[j], (n, 1))
        sigmas[j] = np.dot(np.multiply(row.reshape(n, 1), dif).T, dif) / s

        ps[j] = s / n

    log_vector = np.log(np.array([np.dot(ps.T, ws[:, i]) for i in np.arange(n)]))
    ll_new = np.dot(log_vector.T, np.ones(n))

    print('m: ', ms)
    print('cov: ')
    print(sigmas)
    print('likelihood: ')
    print(ll_new)
    if np.abs(ll_new - ll_old) < 0.1:
        break
    ll_old = ll_new

