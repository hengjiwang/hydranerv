import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.neuron import Neuron

class KENeuron(Neuron):
    """a sigma_w dependent k_e neuron model"""
    def __init__(self, dt=.01, tmax=1000, wnoise=0, ispacemaker=True, t_ref=.1, s=.00277):
        """constructor"""
        super().__init__(dt, tmax, wnoise, ispacemaker, t_ref)
        self.k_e = None
        self.s = s

    def get_k_e(self):
        # return self.sigma_w() ** 3 / 30 / (29000)**2
        return self.sigma_w() / 40

    def step(self, i_ex=0, mech_stim=0):
        """step function"""
        v = self.v()
        sigma_a = self.sigma_a()
        sigma_w = self.sigma_w()

        if self.t < self.t_last + self.t_pk:

            v = self.v_spike
            da = - self.sigma_a() / self.tau_p
            dw = self.k_in
            sigma_a += da * self.dt
            sigma_w += dw * self.dt

        elif self.t < self.t_last + self.t_pk + self.t_ref or v == self.v_spike:

            v = self.v_r
            da = - self.sigma_a() / self.tau_p
            dw = self.k_in
            sigma_a += da * self.dt
            sigma_w += dw * self.dt

        else:
            # Derivatives
            dv = 1 / self.c_m * (- self.i_l() - self.i_s(mech_stim) + i_ex)
            da = - self.sigma_a() / self.tau_p
            dw = self.k_in

            # Update new values
            v += dv * self.dt
            sigma_a += da * self.dt
            sigma_w += dw * self.dt

            if v > self.v_th:
                v = self.v_spike
                sigma_a += self.k_a
                sigma_w += - self.get_k_e()
                self.spikes.append(self.t)
                self.t_last = self.t

        # Update neuron state
        self.v_train.append(v)
        self.sigma_a_train.append(sigma_a)
        self.sigma_w_train.append(sigma_w)
        self.t += self.dt


if __name__ == '__main__':
    nrn = KENeuron()
    nrn.run()
    nrn.disp(figsize=(10,5), disp_stress=True)
