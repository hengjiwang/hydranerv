import numpy as np
from pylab import *

#1) Initialize parameters.
tmax = 600
dt = 0.01

#1.1) Neuron/Network pairs.
c_m = .05
r_m = 20
v_th = -50
v_r = -60

p_th = .5
k_p = .4 # 1
tau_p = 5

k_in = .005
k_e = .04
beta = .2


#2) Reserve memory
T = int(np.ceil(tmax / dt))
v = np.zeros(T)
p = np.zeros(T)
f = np.zeros(T)

v[0] = -60 #Resting potential
p[0] = 0 #Steady state
f[0] = 0

#3) For-loop over time.
for t in np.arange(T-1):
    if v[t] < v_th:
        #3.1) Update DOE.
        if p[t] > p_th:
            dv = 1/c_m * (k_p * (p[t] - p_th) - 1/r_m *(v[t] + 60))
        else:
            dv = 1/c_m * (- 1/r_m *(v[t] + 60))
        v[t+1] = v[t]+dv*dt
        df = - f[t] / tau_p
        f[t+1] = f[t] + df*dt
        dp = k_in + df
        p[t+1] = p[t] + dt*dp
    else:
        #3.2) Spike!
        v[t] = 20
        v[t+1] = v_r
        f[t+1] = f[t] + beta
        p[t+1] = p[t] + beta - k_e

#4) Plot voltage trace
figure(figsize=(10,5))
tvec = np.arange(0, tmax, dt)
plot(tvec, v, 'b', label='v')
# plot(tvec, p, 'r', label='p')
# plot(tvec, f, 'g', label='f')
xlabel('Time[ms]')
xlim(300, 600)
ylabel('Membrane voltage [mV]')
# title('A single qIF neuron with current step input')
legend()
show()