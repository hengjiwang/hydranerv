import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from hydranerv.utils import utils

class TypeIHHNeuron:
    """A modified Hodgkin-Huxley Neuron in Type-I regime (https://neuronaldynamics.epfl.ch/online/Ch2.S3.html#SS2)"""
    def __init__(self):
        pass