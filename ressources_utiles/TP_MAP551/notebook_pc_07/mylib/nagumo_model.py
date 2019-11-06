import numpy as np
import ctypes

def make_nd_array(c_pointer, shape, dtype=np.float64, order='C', own_data=True):
    """ Convert arrayx from C to python """
    arr_size = np.prod(shape[:]) * np.dtype(dtype).itemsize
    #if sys.version_info.major >= 3:
    buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
    buf_from_mem.restype = ctypes.py_object
    buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    buffer = buf_from_mem(c_pointer, arr_size, 0x100)
    #else:
    #    buf_from_mem = ctypes.pythonapi.PyBuffer_FromMemory
    #    buf_from_mem.restype = ctypes.py_object
    #    buffer = buf_from_mem(c_pointer, arr_size)
    arr = np.ndarray(tuple(shape[:]), dtype, buffer, order=order)
    if own_data and not arr.flags.owndata:
        return arr.copy()
    else:
        return arr

class nagumo_model(object): 
 
    def __init__(self, k, d, xmin, xmax, nx) : 
        self.k = k
        self.d = d
        self.xmin = xmin 
        self.xmax = xmax
        self.nx = nx 
        self.dx = (xmax-xmin)/(nx+1) 
 
    def fcn(self, t, y):
        k = self.k
        d = self.d
        nx = self.nx
        dx = self.dx
        doverdxdx = d/(dx*dx)

        ydot = np.zeros(y.size)

        ydot[0] = doverdxdx*(-y[0] + y[1]) + k*y[0]*y[0]*(1 - y[0])
#        for ix in range(1, nx-1):
#            ydot[ix] = doverdxdx*(y[ix-1] - 2*y[ix] + y[ix+1]) + k*y[ix]*y[ix]*(1 - y[ix])
        ydot[1:nx-1] = doverdxdx*(y[0:nx-2] - 2*y[1:nx-1] + y[2:nx]) + k*y[1:nx-1]*y[1:nx-1]*(1 - y[1:nx-1])
        ydot[nx-1] = doverdxdx*(y[nx-2] - y[nx-1]) + k*y[nx-1]*y[nx-1]*(1 - y[nx-1])

        return ydot

    def fcn_radau(self, n, t, y, ydot, rpar, ipar):
        k = self.k
        d = self.d
        nx = self.nx
        dx = self.dx
        doverdxdx = d/(dx*dx)

        #ydot[0] = doverdxdx*(-y[0] + y[1]) + k*y[0]*y[0]*(1 - y[0])
        #for ix in range(1, nx-1):
        #    ydot[ix] = doverdxdx*(y[ix-1] - 2*y[ix] + y[ix+1]) + k*y[ix]*y[ix]*(1 - y[ix])
        #ydot[nx-1] = doverdxdx*(y[nx-2] - y[nx-1]) + k*y[nx-1]*y[nx-1]*(1 - y[nx-1])
        n_python = self.nx
        t_python = make_nd_array(t, (1,))[0]
        y_python = make_nd_array(y, (n_python,))
        y_dot_python = self.fcn(t_python, y_python)
        for i in range(y_dot_python.size):
             ydot[i] = y_dot_python[i]

    def fcn_radau_old(self, n, t, y, ydot, rpar, ipar):
        k = self.k
        d = self.d
        nx = self.nx
        dx = self.dx
        doverdxdx = d/(dx*dx)

        ydot[0] = doverdxdx*(-y[0] + y[1]) + k*y[0]*y[0]*(1 - y[0])
        for ix in range(1, nx-1):
            ydot[ix] = doverdxdx*(y[ix-1] - 2*y[ix] + y[ix+1]) + k*y[ix]*y[ix]*(1 - y[ix])
        ydot[nx-1] = doverdxdx*(y[nx-2] - y[nx-1]) + k*y[nx-1]*y[nx-1]*(1 - y[nx-1])

    def fcn_reac_radau(self, n, t, y, ydot, rpar, ipar):
        k = self.k
        for ix in range(n[0]):
            ydot[ix] = k*y[ix]*y[ix]*(1 - y[ix])
        ydot[0:n[0]] = k*y*y*(1 - y)

    def fcn_diff_radau(self, n, t, y, ydot, rpar, ipar):
        k = self.k
        d = self.d
        nx = self.nx
        dx = self.dx
        doverdxdx = d/(dx*dx)

        ydot[0] = doverdxdx*(-y[0] + y[1])
        for ix in range(1, nx-1):
            ydot[ix] = doverdxdx*(y[ix-1] - 2*y[ix] + y[ix+1])
        ydot[nx-1] = doverdxdx*(y[nx-2] - y[nx-1])

    def fcn_rock(self, n, t, y, ydot):
        k = self.k
        d = self.d
        nx = self.nx
        dx = self.dx
        doverdxdx = d/(dx*dx)

        ydot[0] = doverdxdx*(-y[0] + y[1]) + k*y[0]*y[0]*(1 - y[0])
        for ix in range(1, nx-1):
            ydot[ix] = doverdxdx*(y[ix-1] - 2*y[ix] + y[ix+1]) + k*y[ix]*y[ix]*(1 - y[ix])
        ydot[nx-1] = doverdxdx*(y[nx-2] - y[nx-1]) + k*y[nx-1]*y[nx-1]*(1 - y[nx-1])

    def fcn_diff_rock(self, n, t, y, ydot):
        d = self.d
        nx = self.nx
        dx = self.dx
        doverdxdx = d/(dx*dx)

        ydot[0] = doverdxdx*(-y[0] + y[1]) 
        for ix in range(1, nx-1):
            ydot[ix] = doverdxdx*(y[ix-1] - 2*y[ix] + y[ix+1])
        ydot[nx-1] = doverdxdx*(y[nx-2] - y[nx-1])

    def fcn_exact(self, t):
        k = self.k
        d = self.d
        xmin = self.xmin
        xmax = self.xmax
        nx = self.nx
        dx = self.dx
        x0 = -20.

        v = (1./np.sqrt(2.))*(np.sqrt(k*d))
        cst  = -(1./np.sqrt(2.))*(np.sqrt(k/d))

        x = np.linspace(xmin+dx, xmax-dx, nx)
        y = np.exp(cst*(x-x0-v*t)) / (1. + np.exp(cst*(x-x0-v*t)))
        return y
