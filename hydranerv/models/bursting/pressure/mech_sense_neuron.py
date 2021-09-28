import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.bursting.pressure.lif_neuron import LIFNeuron

class MechSenseNeuron(LIFNeuron):
    """A modified LIF neuron with mechanosensitive ion channel"""
    def __init__(self):
        super().__init__()
        self.p_th = 1 # Pa
        self.k_p = 5 # nA/Pa
        self.r_p_inc = 0.01 # Pa/s
        self.r_p_dec = 0.002 # Pa/s
        self.tau_p = 60 # s
        self.reset()

    def reset(self):
        """reset neuron states"""
        super().reset()
        self.acc_train = [0]
        self.p_train = [0.9] # Pa

    def i_ext(self, p):
        """mechanosensitive channel current"""
        if p > self.p_th:
            return self.k_p * (p - self.p_th)
        else:
            return 0

    def step(self):
        """step function"""

        v = self.v_train[-1]
        p = self.p_train[-1]
        i_m = self.i_ext_train[-1]

        # Update potential
        v = self.update_v(v, i_m)
        self.v_train.append(v)

        # Update spikes effect
        acc = 0
        for t_spike in self.spike_train:
            acc += np.exp( - (self.t - t_spike) / self.tau_p)
        self.acc_train.append(acc)

        # Update pressure
        p += self.dt * (self.r_p_inc - self.r_p_dec * acc)
        self.p_train.append(p)

        # Update mechanosensitive current
        self.i_ext_train.append(self.i_ext(p))

        # Update time
        self.t += self.dt

    def disp(self):
        """display simulation results"""
        time_axis = np.arange(0, len(self.v_train) * self.dt, self.dt)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6))
        ax1.plot(time_axis, self.v_train)
        ax1.set_ylabel('Potential (mV)')
        ax2.plot(time_axis, self.acc_train)
        ax2.set_ylabel('Spikes effects')
        ax3.plot(time_axis, self.p_train)
        ax3.set_ylabel('Pressure (Pa)')
        ax4.plot(time_axis, self.i_ext_train)
        ax4.set_ylabel('Current (nA)')
        ax4.set_xlabel('Time (s)')
        plt.show()

if __name__ == '__main__':
    neuron = MechSenseNeuron()
    neuron.run(200)
    neuron.disp()


