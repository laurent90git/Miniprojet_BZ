import numpy as np

class heat_model:

    def __init__(self, d, xmin, xmax, nx) :
        self.d = d
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.dx = (xmax-xmin)/(nx+1)

    def mat(self):
        d = self.d
        nx = self.nx
        dx = self.dx
        doverdxdx = d/(dx*dx)

        a = np.zeros((nx, nx))

        a[0,0] = -doverdxdx 
        a[0,1] = doverdxdx
        for ix in range(1, nx-1):
            a[ix,ix-1] = doverdxdx 
            a[ix,ix] = -2*doverdxdx 
            a[ix,ix+1] = doverdxdx 
        a[nx-1,nx-2] = doverdxdx 
        a[nx-1,nx-1] = -doverdxdx 

        return a

    def fcn(self, t, y):
        nx = self.nx
        dx = self.dx
        doverdxdx = 1/(dx*dx)

        ydot = np.zeros(y.size)

        ydot[0] = doverdxdx * (-y[0] + y[1])
        for ix in range(1, nx-1):
            ydot[ix] = doverdxdx * (y[ix-1] - 2*y[ix] + y[ix+1])
        ydot[nx-1] = doverdxdx * (y[nx-2] - y[nx-1])

        return ydot

    def fcn_radau(self, n, t, y, ydot, rpar, ipar):
        nx = self.nx
        dx = self.dx
        doverdxdx = 1/(dx*dx)

        ydot[0] = doverdxdx * (-y[0] + y[1])
        for ix in range(1, nx-1):
            ydot[ix] = doverdxdx * (y[ix-1] - 2*y[ix] + y[ix+1])
        ydot[nx-1] = doverdxdx * (y[nx-2] - y[nx-1])

    def fcn_rock(self, n, t, y, ydot):
        nx = self.nx
        dx = self.dx
        doverdxdx = 1/(dx*dx)

        ydot[0] = doverdxdx * (-y[0] + y[1])
        for ix in range(1, nx-1):
            ydot[ix] = doverdxdx * (y[ix-1] - 2*y[ix] + y[ix+1])
        ydot[nx-1] = doverdxdx * (y[nx-2] - y[nx-1])

    def fcn_exact(self, t):
        xmin = self.xmin
        xmax = self.xmax
        nx = self.nx
        dx = self.dx
        x = np.linspace(xmin+dx, xmax-dx, nx)
        y = (1/(2*np.sqrt(np.pi*t))) * np.exp(-(x*x)/(4.*t))
        return y
