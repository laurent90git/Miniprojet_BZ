import numpy as np

class brusselator_model:

    def __init__(self, a, b) :
        self.a = a
        self.b = b

    def fcn(self, t, y):
        y1, y2 = y
        a = self.a
        b = self.b
        y1_dot = 1 - (b+1)*y1 + a*y1*y1*y2
        y2_dot = b*y1 - a*y1*y1*y2
        return np.array([y1_dot, y2_dot])

    def jac(self, t, y):
        y1, y2 = y
        a = self.a
        b = self.b
        return np.array([[2*a*y1*y2 -(b+1), a*y1*y1], [-2*a*y1*y2 + b, -a*y1*y1]])

class oregonator_model:

    def __init__(self, eps, mu, f, q):
        self.eps = eps
        self.mu = mu
        self.f = f
        self.q = q

    def fcn(self, t, y):
        y1, y2, y3 = y
        eps = self.eps
        mu = self.mu
        f = self.f
        q = self.q
        y1_dot = y2 - y1
        y2_dot = (1/eps)*(q*y3 - y3*y2 + y2*(1-y2))
        y3_dot = (1/mu)*(-q*y3 - y3*y2 + f*y1)
        return np.array([y1_dot, y2_dot, y3_dot])

    def jac(self, t, y):
        y1, y2, y3 = y
        eps = self.eps
        mu = self.mu
        f = self.f
        q = self.q
        return np.array([[-1   , 1                   , 0             ],
                         [0    , (1/eps)*(-y3+1-2*y2), (1/eps)*(q-y2)],
                         [ f/mu, -y3/mu              , (1/mu)*(-q-y2)]])
