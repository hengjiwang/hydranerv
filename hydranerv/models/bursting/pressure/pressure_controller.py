import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.bursting.pressure.mech_sense_neuron import MechSenseNeuron

class PressureController:
    """A controller of pressure"""
    def __init__(self, neuron):
        self.neuron = neuron
        self.dt = self.neuron.dt
        self.r_p_inc = .1 # Pa/s
        self.r_d = .1 # Pa/s
        self.r_h = .05 # Pa/s
        self.tau_d = 2 # s
        self.tau_h = 10 # s
        self.reset()

    def reset(self):
        self.p_train = [0.9]
        self.acc_train = [0]

    def t(self):
        return self.neuron.t

    def acc(self):
        return self.acc_train[-1]

    def pressure(self):
        return self.p_train[-1]

    def update_acc(self):
        """update spikes effect"""
        acc = 0
        for t_spike in self.neuron.spike_train:
            acc += self.r_d * np.exp( - (self.t() - t_spike) / self.tau_d)
            acc += - self.r_h * np.exp( - (self.t() - t_spike) / self.tau_h)
        self.acc_train.append(acc)

    def update_p(self):
        """update pressure"""
        p = self.pressure()
        acc = self.acc()
        p = p + self.dt * (self.r_p_inc + acc)
        self.p_train.append(p)

    def update(self):
        self.update_acc()
        self.update_p()

    def disp(self):
        time_axis = np.arange(self.dt, len(self.p_train) * self.dt, self.dt)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        ax1.plot(time_axis, self.acc_train[1:])
        ax1.set_ylabel('Spikes effects')
        ax2.plot(time_axis, self.p_train[1:])
        ax2.set_ylabel('Pressure (Pa)')
        ax2.set_xlabel('Time (s)')
        plt.show()


if __name__ == '__main__':
    dt = 0.02
    t_total = 200
    neuron = MechSenseNeuron(dt)
    pcontroller = PressureController(neuron)
    time_axis = np.arange(dt, t_total, dt)
    for t in time_axis:
        neuron.step(p=pcontroller.pressure())
        pcontroller.update()
    neuron.disp()
    pcontroller.disp()


