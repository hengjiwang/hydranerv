import numpy as np
from hydranerv.model.lifneuron import LIFNeuron


class CBPacemaker(LIFNeuron):

    def __init__(self):
        super().__init__()

    def _theta_to_i_ext(self, theta):
        """Override"""
        return 60 / (1 + np.exp(10*(theta - 4)))

    def _light_to_i_ext(self, light):
        """Override"""
        return -10 * light * self.v

