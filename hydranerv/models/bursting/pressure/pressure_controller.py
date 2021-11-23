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
        self.tau_p = 5
        self.reset()

    def reset(self):
        self.p_train = [0]
        self.vsize_train = [0]
        self.p_ext_train = [0]

    def t(self):
        return self.neuron.t

    def vsize(self):
        return self.vsize_train[-1]

    def p_ext(self):
        return self.p_ext_train[-1]

    def pressure(self):
        return self.p_train[-1]

    def update_vsize(self):
        """update spikes effect"""
        if self.neuron.spike_train and self.neuron.spike_train[-1] == self.t() - self.dt:
            self.vsize_train.append(self.vsize() - self.k_e)
        else:
            self.vsize_train.append(self.vsize() + self.dt * self.k_in)

    def update_p_ext(self):
        """update external pressure"""
        if self.neuron.spike_train and self.neuron.spike_train[-1] == self.t() - self.dt:
            self.p_ext_train.append(self.p_ext() + self.beta)
        else:
            self.p_ext_train.append(self.p_ext() - self.dt * self.p_ext() / self.tau_p)

    def update_p(self):
        """update pressure"""
        p = self.alpha * self.vsize() + self.p_ext()
        self.p_train.append(p)

    def update(self):
        self.update_vsize()
        self.update_p_ext()
        self.update_p()

    def disp(self, xmin=None, xmax=None):
        time_axis = np.arange(self.dt, len(self.p_train) * self.dt, self.dt)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))
        ax1.plot(time_axis, self.neuron.v_train[1:], 'b')
        ax1.set_ylabel('potential (mV)')
        ax2.plot(time_axis, self.vsize_train[1:], 'g', label='vacuole size')
        ax2.plot(time_axis, self.p_ext_train[1:], 'r', label='external')
        ax2.plot(time_axis, self.p_train[1:], 'k', label='pressure')
        ax2.legend()
        ax2.set_ylabel('driving terms')
        ax3.plot(time_axis, self.neuron.i_mem_train[1:], 'gray')
        ax3.set_ylabel('I_mem (nA)')
        ax3.set_xlabel('Time (s)')
        if xmin and xmax:
            ax1.set_xlim(xmin, xmax)
            ax2.set_xlim(xmin, xmax)
            ax3.set_xlim(xmin, xmax)
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


