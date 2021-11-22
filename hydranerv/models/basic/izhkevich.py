# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:47:11 2016

@author: arthur

Izhikevich neuron - Original.
"""
import numpy as np
from pylab import *

#1) Initialize parameters.
tmax = 1000
dt = 0.5

#1.1) Neuron/Network pairs.
a = 0.02
b = 0.2
c = -50
d = 2

#1.2) Input pairs
lapp = 10
tr = np.array([200, 700])/dt  #stm time

#2) Reserve memory
T = int(np.ceil(tmax / dt))
v = np.zeros(T)
u = np.zeros(T)
d_v = np.zeros(T)

v[0] = -70 #Resting potential
u[0] = -14 #Steady state
d_v[0] = 0

#3) For-loop over time.
for t in np.arange(T-1):
#3.1) Get input.
    if t > tr[0] and t < tr[1]:
        l = lapp
    else:
        l = 0
    if v[t] < 35:
        #3.2) Update DOE.
        dv = (0.04*v[t]+5)*v[t]+140-u[t]
        d_v[t+1] = dv + l
        v[t+1] = v[t]+(dv+l)*dt
        du = a*(b*v[t]-u[t])
        u[t+1] = u[t] + dt*du
    else:
        #3.3) Spike!
        v[t] = 35
        v[t+1] = c
        u[t+1] = u[t] + d
        d_v[t+1] = d_v[t]

#4) Plot voltage trace
figure()
tvec = np.arange(0, tmax, dt)
plot(tvec, v, 'b', label='u')
plot(tvec, u, 'r', label='w')
# plot(tvec, d_v, 'g', label='dv')
xlabel('Time[ms]')
ylabel('Membrane voltage [mV]')
title('A single qIF neuron with current step input')
legend()
show()