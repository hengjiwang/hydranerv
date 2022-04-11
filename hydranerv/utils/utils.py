import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, os
from collections import defaultdict
from . import disp


def euler_odeint(rhs, y, T, dt, **kwargs):
    """An Euler method integrator"""
    sol = np.zeros((int(T/dt), len(y)))

    for j in np.arange(0, int(T/dt)):
        sol[j, :] = y
        t = j * dt
        dydt = rhs(y, t, **kwargs)
        y += dydt * dt

    return sol


def run_neuron(T, dt, i_ext_train, neuron):
    time = np.arange(0, T + dt, dt)
    for j, t in enumerate(time):
        i_ext = i_ext_train[j]
        neuron.step(t, 0, 0, 0, i_ext)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(time / 1000, neuron.v_train, 'b')
    ax1.set_xlabel('t (s)')
    ax1.set_ylabel('v')
    ax2 = ax1.twinx()
    ax2.plot(time / 1000, i_ext_train, 'r')
    ax2.set_ylabel('I$_{ext}$')
    ax2.set_ylim(0, 100)
    plt.show()

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


def euclid_dist(pt1, pt2):
    """Calculates the Euclidean distance between pt1 and pt2"""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def std_euclid_dist(pt1, pt2, std=(1, 1)):
    """Calculates the standardized Euclidean distance between pt1 and pt2"""
    return np.sqrt(((pt1[0] - pt2[0]) / std[0])**2 + ((pt1[1] - pt2[1]) / std[1])**2)

def cyl_dist(pt1, pt2):
    """Calculates the distance between pt1 and pt2 on a cylindrical surface"""
    r1, phi1, z1 = pt1
    r2, phi2, z2 = pt2

    if r1 != r2:
        raise Exception('pt1 and pt2 must be on the same cylindrical surface!')

    phi_diff = abs(phi2 - phi1)
    phi_diff = phi_diff if phi_diff < np.pi else 2 * np.pi - phi_diff

    return np.sqrt((r1 * (phi_diff)) ** 2 + (z2 - z1) ** 2)


def min_max_norm(l, rescale=1, offset=0):
    minv, maxv = min(l), max(l)
    if minv == maxv:
        return [0 for _ in l]
    return [rescale * (x - minv) / (maxv - minv) + offset for x in l]

# def cluster_spikes(spike_train, mul=5):
#     """Separate spikes to clusters and return a dictionary"""
#     res = defaultdict(list)
#     n = len(spike_train)
#     if n == 0:
#         return res
#     if n <= 2:
#         res[0] = spike_train
#         return res

#     typical_isi = spike_train[1] - spike_train[0]
#     j = 0
#     res[0].append(spike_train[0])
#     for i in range(1, n):
#         if spike_train[i] - spike_train[i-1] >= typical_isi * mul:
#             j += 1
#         res[j].append(spike_train[i])
#     return res

def cluster_peaks(peaks, min_cb_interval, realign=True):
    """Separate peaks into different clusters based on min_cb_interval(in frame numbers)"""
    clusters = [[]]

    # Clustering peaks
    for j in range(len(peaks)-1):
        pk = peaks[j]
        pk_nxt = peaks[j+1]
        clusters[-1].append(pk)
        if pk_nxt - pk < min_cb_interval:
            pass
        else:
            clusters.append([])

    clusters[-1].append(peaks[-1])

    # Subtracting offsets
    indices_to_keep = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        if len(cluster) >= 2:
            indices_to_keep.append(i)
        if realign:
            offset = cluster[0]
            for j in range(len(cluster)):
                cluster[j] -= offset

    return np.array(clusters, dtype=list)[indices_to_keep]


def transpose_2d_list(list2d):
    """Returns a transposed 2d list"""

    maxlen = max([len(l) for l in list2d])
    for j in range(len(list2d)):
        list2d[j].extend([None]*(maxlen-len(list2d[j])))

    list2d = np.array(list2d).T.tolist()

    for j in range(len(list2d)):
        list2d[j] = [x for x in list2d[j] if x]

    return list2d

def get_clusters(nmovie, fpath='./cb_locs/wataru_data/ctr/', display=True, offsets=None, realign=True):
    if display:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
    inc = 0
    # Read txt files
    filelist = sorted(os.listdir(fpath))
    clusters_all = []
    clusters_per_video = []

    for imovie in range(1, nmovie+1):

        # Extract peak locations
        locs = []
        offset = 0
        for filename in filelist:
            if filename.endswith(".txt") and filename.startswith(str(imovie)):
                if offsets:
                    offset = offsets[imovie-1][int(filename.split('_cb_locs')[0][-1])-1]
                locs.append(pd.read_csv(fpath + filename, header=None).values[0] + offset)

        # Cluster the peaks
        clusters = []
        for peaks in locs:
            clusters.extend(list(cluster_peaks(peaks, 50, realign=realign)))

        # Plot the cluster spikes
        if display:
            disp.add_spike_trains(ax, clusters, 2, lw=1, color=randomcolor(), inc=inc)
        clusters_all.extend(clusters)
        clusters_per_video.append(clusters)
        inc += len(clusters)

    if display:
        plt.xlim(0, 300)
        plt.ylim(0, inc)
        plt.show()
    return clusters_all, clusters_per_video