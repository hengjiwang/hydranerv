import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.bursting.pressure.mech_sense_neuron import MechSenseNeuron

class PressureController:
    """A controller of pressure"""
    def __init__(self, neuron):
        self.neuron = neuron
        self.dt = self.neuron.dt
        self.k_in = .005
        self.k_e = .04
        self.alpha = 1
        self.beta = .2
        self.tau_c = 5
        self.reset()

    def reset(self):
        self.p_train = [0]
        self.vsize_train = [0]

    def t(self):
        return self.neuron.t

    def vsize(self):
        return self.vsize_train[-1]

    def pressure(self):
        return self.p_train[-1]

    def update_vsize(self):
        """update spikes effect"""
        vsize = self.k_in * self.t()
        for t_spike in self.neuron.spike_train:
            vsize -= self.k_e
        self.vsize_train.append(vsize)

    def update_p(self):
        """update pressure"""
        p = self.alpha * self.vsize()
        for t_spike in self.neuron.spike_train:
            p += self.beta * np.exp(- (self.t() - t_spike) / self.tau_c)
        self.p_train.append(p)

    def update(self):
        self.update_vsize()
        self.update_p()

    def disp(self, xmin=None, xmax=None):
        time_axis = np.arange(self.dt, len(self.p_train) * self.dt, self.dt)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
        ax1.plot(time_axis, self.neuron.v_train[1:])
        ax1.set_ylabel('Potential (mV)')
        ax2.plot(time_axis, self.vsize_train[1:])
        ax2.set_ylabel('Vacuole size')
        ax3.plot(time_axis, self.p_train[1:])
        ax3.set_ylabel('Pressure (Pa)')
        ax4.plot(time_axis, self.neuron.i_mem_train[1:])
        ax4.set_ylabel('I_mem (nA)')
        ax4.set_xlabel('Time (s)')
        if xmin and xmax:
            ax1.set_xlim(xmin, xmax)
            ax2.set_xlim(xmin, xmax)
            ax3.set_xlim(xmin, xmax)
            ax4.set_xlim(xmin, xmax)
        plt.show()


if __name__ == '__main__':
    dt = 0.02
    t_total = 600
    neuron = MechSenseNeuron(dt)
    pcontroller = PressureController(neuron)
    time_axis = np.arange(dt, t_total, dt)
    for t in time_axis:
        neuron.step(p=pcontroller.pressure())
        pcontroller.update()
    pcontroller.disp(xmin=150, xmax=400)


