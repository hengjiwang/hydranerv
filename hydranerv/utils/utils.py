import numpy as np
import matplotlib.pyplot as plt
import random


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

