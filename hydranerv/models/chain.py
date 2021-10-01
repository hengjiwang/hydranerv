import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from hydranerv.models.bursting.pressure.mech_sense_neuron import MechSenseNeuron

class Chain:
    """A chain (1D) of gap-junctional coupled neurons"""
    def __init__(self, neuron_type, num=10):
        pass