import numpy as np
from collections import deque

class LatentVariable:

    def __init__(self):
        self.dt = 1
        self.theta = 5
        self.cb_fire_records = deque()
        self.theta_train = []
        self.dec_ratio = 0.000005 # 0.000025
        self.inc_ratio = 0.000005
        self.dur_valid = 50 * 1000
        self.deriv_train = []

    def rhs(self, t):
        """Derivative of theta"""

        deriv = -self.dec_ratio * self.theta
        while self.cb_fire_records:
            t0 = self.cb_fire_records.popleft()
            if t - t0 > self.dur_valid:
                continue
            else:
                self.cb_fire_records.appendleft(t0)
                deriv += len(self.cb_fire_records) * self.inc_ratio
                break

        self.deriv_train.append(deriv)
        return deriv

    def step(self, t, cb_fired=False):
        """Step function"""

        if cb_fired:
            self.cb_fire_records.append(t)
        self.theta = self.theta + self.rhs(t) * self.dt
        self.theta_train.append(self.theta)

        return self.theta
