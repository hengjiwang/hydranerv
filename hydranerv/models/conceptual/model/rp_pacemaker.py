from hydranerv.models.basic.lif_neuron import LIFNeuron


class RPPacemaker(LIFNeuron):
    def __init__(self, theta_amp):
        super().__init__()
        self.theta_amp = theta_amp

    def _theta_to_i_ext(self, theta):
        """Overload"""
        return self.theta_amp * theta

    def _light_to_i_ext(self, light):
        """Overload"""
        return light
