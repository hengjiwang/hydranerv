import numpy as np
from hydranerv.model.lifneuron import LIFNeuron


class CBPacemaker(LIFNeuron):

    def __init__(self, theta_amp=60):
        self.theta_amp = theta_amp
        super().__init__()

    def _theta_to_i_ext(self, theta):
        """Override"""
        return self.theta_amp / (1 + np.exp(10*(theta - 4)))

    def _light_to_i_ext(self, light):
        """Override"""
        return -10 * light * self.v

