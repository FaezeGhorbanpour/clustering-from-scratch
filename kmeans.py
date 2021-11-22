import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

college = pd.read_csv('College.csv').values
college_ = college[:, 2:]


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


import matplotlib.pyplot as plt
import numpy as np

ks = [i for i in range(1, 5)]
errors = list()
for k in ks:
    x, y, z = kmeans(k, 3)
    errors.append(z)

plt.gca().set_color_cycle(['red'])
plt.plot(ks, errors)
plt.legend(['error'], loc='upper left')
plt.show()
