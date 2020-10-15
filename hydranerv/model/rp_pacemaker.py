from hydranerv.model.lifneuron import LIFNeuron


class RPPacemaker(LIFNeuron):
    def __init__(self):
        super().__init__()

    def _theta_to_i_ext(self, theta):
        """Overload"""
        return theta

    def _light_to_i_ext(self, light):
        """Overload"""
        return light
