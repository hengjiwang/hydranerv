import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from hydranerv.models.basic.lif_neuron import LIFNeuron
from hydranerv.models.conceptual.model.cb_pacemaker import CBPacemaker
from hydranerv.models.conceptual.model.rp_pacemaker import RPPacemaker
import hydranerv.utils.utils as utils


class Network:

    def __init__(self, type, numx, numy, neuron_density, pacemaker_density, link_maxdist, link_density, gc, theta_amp):
        self.neurons = defaultdict(LIFNeuron)
        self.neighbors = defaultdict(set)
        self.edges = set()
        self.type = type
        self.numx = numx
        self.numy = numy
        self.neuron_density = neuron_density
        self.pacemaker_density = pacemaker_density
        self.link_maxdist = link_maxdist
        self.link_density = link_density
        self.gc = gc
        self.theta_amp = theta_amp
        self.generate_neurons()
        self.generate_links()

    def generate_neurons(self):
        """Generate neurons in the network"""
        for y in range(self.numy):
            for x in range(self.numx):
                if y == 0 and self.type == "CB":
                    self.neurons[(x, y)] = LIFNeuron()
                elif y < self.numy // 10 and self.type == "RP":
                    if np.random.uniform() < self.neuron_density:
                        if np.random.uniform() < self.pacemaker_density:
                            self.neurons[(x, y)] = RPPacemaker(self.theta_amp)
                        else:
                            self.neurons[(x, y)] = LIFNeuron()
                elif y > self.numy // 10 * 9 and self.type == "CB":
                    if np.random.uniform() < self.neuron_density:
                        if np.random.uniform() < self.pacemaker_density:
                            self.neurons[(x, y)] = CBPacemaker(self.theta_amp)
                        else:
                            self.neurons[(x, y)] = LIFNeuron()
                elif np.random.uniform() < self.neuron_density:
                    self.neurons[(x, y)] = LIFNeuron()

    def generate_links(self, structure="MST", dist=utils.std_euclid_dist, std=(1, 4)):
        """Generate links between neurons based on specified algorithm"""
        if structure == "MST":
            self._mst(dist, std)

    def _mst(self, dist, std):
        """Construct the minimum spanning tree based on Kruskal algorithm"""
        # Construct fully connected graph
        edges = set()
        neurons = list(self.neurons.keys())

        for i in range(len(neurons)):
            for j in range(len(neurons)):
                n1 = neurons[i]
                n2 = neurons[j]
                if n1 != n2:
                    n1, n2 = sorted([n1, n2])
                    edges.add((dist(n1, n2, std), n1, n2))

        # Initiate subtrees
        subtree = defaultdict()
        for n in self.neurons:
            subtree[n] = n

        # Sort the edges based on weights
        edges = sorted(list(edges))

        # Build the MST
        for edge in edges:
            n1 = edge[1]
            n2 = edge[2]

            # Find the root of n1
            root1, d1 = n1, 0
            while subtree[root1] != root1:
                d1 += 1
                root1 = subtree[root1]

            # Find the root of n2
            root2, d2 = n2, 0
            while subtree[root2] != root2:
                d2 += 1
                root2 = subtree[root2]

            # If they are not in the same subtree
            if root1 != root2:
                # Add the edge
                self.edges.add((n1, n2))
                self.neighbors[n1].add(n2)
                self.neighbors[n2].add(n1)

                # Union the two subtrees
                if d1 <= d2:
                    subtree[root2] = root1
                    subtree[n1] = root1
                    subtree[n2] = root1
                else:
                    subtree[root1] = root2
                    subtree[n1] = root2
                    subtree[n2] = root2

    def step(self, t, theta_mat, light_mat, i_stim_mat):
        """Step all neurons of the network"""
        for pos in self.neurons:
            neuron = self.neurons[pos]
            i_syn = 0
            for neighbor in self.neighbors[pos]:
                neuron2 = self.neurons[neighbor]
                i_syn += self.gc * (neuron2.v - neuron.v)
            neuron.step(t, theta_mat[pos[0], pos[1]], light_mat[pos[0], pos[1]], i_syn, i_stim_mat[pos[0], pos[1]])

    def display(self):
        """Display the configuration of the network"""
        plt.figure(figsize=(5, 10))
        plt.xlim(-0.5, self.numx-0.5)
        plt.ylim(-0.5, self.numy-0.5)
        plt.xticks(np.arange(self.numx), rotation=45, fontsize=8)
        plt.yticks(np.arange(self.numy), rotation=0, fontsize=8)
        plt.title(self.type + " Network")
        plt.grid()

        for (x, y) in self.neurons:

            if isinstance(self.neurons[(x, y)], CBPacemaker):
                plt.plot(x, y, color='darkblue', marker='o', mec='gold')
            elif isinstance(self.neurons[(x, y)], RPPacemaker):
                plt.plot(x, y, color='darkgreen', marker='o', mec='gold')
            elif self.type == "CB":
                plt.plot(x, y, color='royalblue', marker='o')
            elif self.type == "RP":
                plt.plot(x, y, color='limegreen', marker='o')

            for (x2, y2) in self.neighbors[(x, y)]:
                if self.type == "CB":
                    plt.plot([x, x2], [y, y2], 'b', alpha=0.2)
                if self.type == "RP":
                    plt.plot([x, x2], [y, y2], 'g', alpha=0.2)

        plt.show()


if __name__ == "__main__":
    network = Network("CB", 30, 60, 0.1, 1, 2, 0.2, 20000, 250)
    network.display()










