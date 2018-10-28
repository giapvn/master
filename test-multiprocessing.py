import multiprocessing as mp
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations
import time
from scipy.stats import multivariate_normal
import timeit

np.random.seed(123)

def parzen_eatimation(x_samples, point_x, h):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        :param x_samples: training sample, 'd x 1'-dimensional numpy array
        :param point_x: point x for density estimation, 'd x 1'-dimensional numpy array
        :param h: window width
        :return: the predicted pdf as float
    """
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:, np.newaxis]) / (h)
        for row in x_i:
            if np.abs(row) > (1/2):
                break
        else: # "completion-else"*
                k_n += 1
    return (h, (k_n / len(x_samples)) / (h**point_x.shape[1]))

def serial(samples, x, widths):
    return [parzen_eatimation(samples, x, w) for w in widths]

def multiprocess(processes, samples, x, widths):
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(parzen_eatimation, args=(samples, x, w)) for w in widths]
    results = [p.get() for p in results]
    results.sort() # to sort the results by input window width
    return results

start = time.time()
fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# Plot Points
# samples within the cube
X_inside = np.array([[0,0,0],[0.2,0.2,0.2],[0.1,-0.1,-0.3]])
X_outside = np.array([[-1.2,0.3,-0.3],[0.8,-0.82,-0.9],[1,0.6,-0.7],[0.8,0.7,0.2],
                      [0.7,-0.8,-0.45],[-0.3,0.6,0.9],[0.7,-0.6,-0.8]])

for row in X_inside:
    ax.scatter(row[0], row[1], row[2], color="r", s=50, marker='^')

for row in X_outside:
    ax.scatter(row[0], row[1], row[2], color="k", s=50)

# Plot cube

point_x = np.array([[0], [0]])

# Generate random 2D-patterns
mu_vec = np.array([0, 0])
cov_mat = np.array([[1, 0], [0, 1]])
n = 10000
x_2Dgauss = np.random.multivariate_normal(mu_vec, cov_mat, n)

var = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
print('actual probability density: ', var.pdf([0, 0]))

widths = np.linspace(1.0, 1.2, 100)
#results = []

#results = multiprocess(8, x_2Dgauss, point_x, widths)

#for r in results:
#    print('h = %s, p(x) = %s' %(r[0], r[1]))

benchmarks = []
benchmarks.append(timeit.Timer('serial(x_2Dgauss, point_x, widths)',
                               'from __main__ import serial, x_2Dgauss, point_x, widths').timeit(number=1))
benchmarks.append(timeit.Timer('multiprocess(2, x_2Dgauss, point_x, widths)',
                               'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
benchmarks.append(timeit.Timer('multiprocess(4, x_2Dgauss, point_x, widths)',
                               'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
benchmarks.append(timeit.Timer('multiprocess(6, x_2Dgauss, point_x, widths)',
                               'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
benchmarks.append(timeit.Timer('multiprocess(8, x_2Dgauss, point_x, widths)',
                               'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
import platform

def print_sysinfo():
    print('\nPython version: ', platform.python_version())
    print('compiler: ', platform.python_compiler())
    print('\nsystem: ', platform.system())
    print('release: ', platform.release())
    print('machine: ', platform.machine())
    print('processor: ', platform.processor())
    print('CPU count: ', mp.cpu_count())
    print('interpreter: ', platform.architecture()[0])

def plot_results():
    bar_labels = ['serial', '2', '4', '6', '8']
    fig = plt.figure(figsize=(10, 8))

    # plot bars
    y_position = np.arange(len(benchmarks))
    plt.yticks(y_position, bar_labels, fontsize=16)
    bars = plt.barh(y_position, benchmarks, align='center', alpha=0.4, color='g')
    # annotation and labels
    for ba, be in zip(bars, benchmarks):
        plt.text(ba.get_width() + 2, ba.get_y() + ba.get_height()/2,
                 '{0:.2%}'.format(benchmarks[0]/be), ha='center', va='bottom', fontsize=12)

    plt.xlabel('time in seconds for n=%s' %n, fontsize=14)
    plt.ylabel('number of processes', fontsize=14)
    t = plt.title('Serial vs. Multiprocessing via Parzen-window estimation', fontsize=18)
    plt.ylim([-1, len(benchmarks)+0.5])
    plt.xlim([0, max(benchmarks)*1.1])
    plt.vlines(benchmarks[0], -1, len(benchmarks)+0.5, linestyles='dashed')
    plt.grid()
    plt.show()

plot_results()
print_sysinfo()

print(time.time()-start)