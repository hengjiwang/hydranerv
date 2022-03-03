from hydranerv.models.bursting.pressure.network_base import NetworkBase

class Chain(NetworkBase):
    """A chain (1D) of gap-junctional coupled neurons"""
    def __init__(self, num=10, gc=100, dt=0.01):
        """constructor"""
        super().__init__(num, gc, dt)

    def make_connections(self):
        self.neighbors[0].append(1)
        self.neighbors[self.num - 1].append(self.num - 2)
        for i in range(1, self.num - 1):
            self.neighbors[i].append(i - 1)
            self.neighbors[i].append(i + 1)

if __name__ == '__main__':
    chain = Chain(gc=.2)
    chain.run(100)
    chain.disp(spike=False)

