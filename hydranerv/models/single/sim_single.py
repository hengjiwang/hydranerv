import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1) Initialize parameters.
tmax = 1000
dt = 0.01

#1.1) Neuron/Network pairs.
c_m = 50 # nF
v_th = -55 # mV
v_r = -75 # mV
g_l = 15 # nS
e_l = v_r # mV
v_spike = 20 # mV

# Bursting parameters
alpha = 1 # Pa/ml
tau_p = 5 # s
k_in = 50 # ml/s

k_a = 5150 # Pa
k_e = 600 # ml


# PIEZO channel parameters
g_s = 25 # nS
e_srev = 10 # mV
s = .00277 # 1/Pa
k_b = 106
m = 25
q = 1

#2) Reserve memory
T = int(np.ceil(tmax / dt))
v = np.zeros(T)
sigma_m = np.zeros(T)
sigma_a = np.zeros(T)
vol = np.zeros(T)
i_s = np.zeros(T)

spikes = []

v[0] = -75 #Resting potential
sigma_a[0] = 0
vol[0] = 25000
sigma_m[0] = sigma_a[0] + alpha * vol[0]

#3) For-loop over time.
for t in np.arange(T-1):
    if v[t] < v_th:
        #3.1) Update DOE.
        i_s[t+1] = g_s / (1 + k_b * np.exp(- s * (sigma_m[t] / m) ** q)) * (v[t] - e_srev)
        i_l = g_l * (v[t] - e_l)
        # update membrane potential
        dv = 1 / c_m * (- i_l - i_s[t])
        v[t+1] = v[t] + dv*dt
        # update active stress
        dsigma_a = - sigma_a[t] / tau_p
        sigma_a[t+1] = sigma_a[t] + dsigma_a * dt
        # update vacuole volume
        dvol = k_in
        vol[t+1] = vol[t] + dvol * dt
    else:
        #3.2) Spike!
        v[t] = 20
        v[t+1] = v_r
        sigma_a[t+1] = sigma_a[t] + k_a
        vol[t+1] = vol[t] - k_e
        i_s[t+1] = i_s[t]
        spikes.append(t*dt)
    # update muscle stress
    sigma_m[t+1] = alpha * vol[t+1] + sigma_a[t+1]

#4) Plot voltage trace
plt.rcParams.update({'font.size': 18})
# fig = plt.figure(figsize=(7,5))
fig = plt.figure(figsize=(7,5))
tvec = np.arange(dt, tmax, dt)
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)
# ax3 = fig.add_subplot(413)
# ax4 = fig.add_subplot(414)
ax1.plot(tvec, v[1:], 'k', linewidth=1)
# ax2.plot(tvec, vol[1:]/1000, 'b', label=r'$\sigma_w$')
# ax2.plot(tvec, sigma_a[1:]/1000, 'g', label=r'$\sigma_a$')
# ax3.plot(tvec, sigma_m[1:], 'k', label=r'$\sigma_m$')
# ax4.plot(tvec, - i_s[1:], 'darkblue', label='Im')
ax1.set_xlabel('Time[s]')
ax1.set_ylabel('V(mV)')
# ax2.set_ylabel('muscle stress (kPa)')
# ax3.set_ylabel(r'$\sigma_m$ (Pa)')
# ax4.set_ylabel(r'$I_s$ (nA)')
ax1.set_title('k_in=' + str(k_in))
# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()
start, end = 550, 950
ax1.set_xlim(start, end)
# ax2.set_xlim(start, end)
# ax3.set_xlim(start, end)
# ax4.set_xlim(start, end)
plt.tight_layout()
plt.show()

# Plot ISI
clusters = []
for i, tspike in enumerate(spikes):
    if i == 0 or tspike - spikes[i-1] > 20:
        clusters.append([tspike])
    else:
        clusters[-1].append(tspike)
plt.figure(figsize=(5, 5))
cluster = clusters[1]
isi = [cluster[i] - cluster[i-1] for i in range(1, len(cluster))]
pd.DataFrame(isi).to_csv('./output/isi_model.csv')
plt.plot(isi, 'k.--', markersize=20)
plt.xlabel('Interval #')
plt.ylabel('ISI[s]')
plt.show()
