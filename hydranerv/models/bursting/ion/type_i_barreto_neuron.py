import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from hydranerv.utils import utils
from hydranerv.models.bursting.ion.barreto_neuron import BarretoNeuron

class TypeIBarretoNeuron(BarretoNeuron):
    """A modified barreto neuron with same sodium channel with type-i hh model"""
    pass
