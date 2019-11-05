import numpy as np
import ctypes as ct


####################################################################
def integrate(method, tini, tend, nt, tol, yini, fcn, fcn_reac, fcn_diff):

    if method=='radau5':
        sol = radau5(tini, tend, yini, fcn, njac=1, rtol=tol, atol=tol, iout=0)
    elif method=='rock4':
        sol = rock4(tini, tend, yini, fcn, rtol=tol, atol=tol)
    elif method=='strang':
        sol = strang(tini, tend, nt, yini, fcn_diff, fcn_reac, tol_diff=1.e-08, tol_reac=1.e-08)
    else:
        print("The integration method " + method + " is unknown for Nagumo model")
        exit(1)
    return sol


#############################################################################
class radau_result:
    def __init__(self, y, t, nfev, njev, nstep, naccpt, nrejct, ndec, nsol):
        self.y = y
        self.t = t
        self.nfev = nfev
        self.njev = njev
        self.nstep = nstep
        self.naccpt = naccpt
        self.nrejct = nrejct
        self.ndec = ndec
        self.nsol = nsol

#############################################################################
def radau5(tini, tend, yini, fcn, njac, rtol=1.e-8, atol=1.e-8, iout=0):

    c_integration = ct.CDLL("./mylib/lib_radau_rock.so")

    tsol=[]
    ysol=[]

    def solout(nr, told, t, y, cont, lrc, n, rpar, ipar, irtrn):
        ##import ipdb; ipdb.set_trace()
        tsol.append(t[0])
        tmp = []
        for i in range(n[0]):
            tmp.append(y[i])
        ysol.append(np.array(tmp))

    fcn_type = ct.CFUNCTYPE(None, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int))
    solout_type = ct.CFUNCTYPE(None, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                               ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int),
                               ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int))

    c_radau5 = c_integration.radau5_integration
    c_radau5.argtypes = [ct.c_double, ct.c_double, ct.c_int,
                         np.ctypeslib.ndpointer(dtype = np.float64),
                         np.ctypeslib.ndpointer(dtype = np.float64),
                         fcn_type, solout_type, ct.c_double, ct.c_double, ct.c_int,
                         ct.c_int, np.ctypeslib.ndpointer(dtype = np.int32)]
    c_radau5.restype = None

    callable_fcn = fcn_type(fcn)
    callable_solout = solout_type(solout)

    yini_array = np.array(yini)
    neq = yini_array.size
    yn = np.zeros(neq)
    info= np.zeros(7, dtype=np.int32)
    c_radau5(tini, tend, neq, yini_array, yn, callable_fcn, callable_solout, rtol, atol, njac, iout, info)

    if iout == 1:
        tsol = np.array(tsol)
        ysol = np.array(np.transpose(np.array(ysol)), order='F')
    else:
        tsol = tend
        ysol = yn

    nfev   = info[0]  # number of function evaluations
    njev   = info[1]  # number of jacobian evaluations
    nstep  = info[2]  # number of computed steps
    naccpt = info[3]  # number of accepted steps
    nrejct = info[4]  # number of rejected steps
    ndec   = info[5]  # number of lu-decompositions
    nsol   = info[6]  # number of forward-backward substitutions

    return radau_result(ysol, tsol, nfev, njev, nstep, naccpt, nrejct, ndec, nsol)

#############################################################################
class rock_result:
    def __init__(self, y, nfev, nstep, naccpt, nrejct, nfevrho, nstage):
        self.y = y
        self.nfev = nfev
        self.nstep = nstep
        self.naccpt = naccpt
        self.nrejct = nrejct
        self.nfevrho = nfevrho
        self.nstage = nstage

#############################################################################
def rock4(tini, tend, yini, fcn, rtol=1.e-8, atol=1.e-8):

    c_integration = ct.CDLL("./mylib/lib_radau_rock.so")

    fcn_type = ct.CFUNCTYPE(None, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                            ct.POINTER(ct.c_double))

    c_rock4 = c_integration.rock4_integration
    c_rock4.argtypes = [ct.c_double, ct.c_double, ct.c_int,
                        np.ctypeslib.ndpointer(dtype=np.float64),
                        np.ctypeslib.ndpointer(dtype=np.float64),
                        fcn_type, ct.c_double, ct.c_double, np.ctypeslib.ndpointer(dtype=np.int32)]
    c_rock4.restype = None

    callable_fcn = fcn_type(fcn)

    neq = yini.size
    y = np.zeros(neq)
    info = np.zeros(8, dtype=np.int32)
    c_rock4(tini, tend, neq, yini, y, callable_fcn, rtol, atol, info)

    nfev    = info[0]   # number of function evaluations.
    nstep   = info[1]   # number of computed steps
    naccpt  = info[2]   # number of accepted steps
    nrejct  = info[3]   # number of rejected steps
    nfevrho = info[4]   # number of evaluations of f used to estimate the spectral radius
    nstage  = info[5]   # maximum number of stages used

    return rock_result(y, nfev, nstep, naccpt, nrejct, nfevrho, nstage)

#############################################################################
class strang_result:
    def __init__(self, y):
        self.y = y

#############################################################################
def strang(tini, tend, nt, yini, fcn_diff, fcn_reac, tol_diff=1.e-12, tol_reac=1.e-12):

    t = np.linspace(tini, tend, nt)
    dt = (tend-tini)/(nt-1)

    ysol = yini 

    for it, ti in enumerate(t[:-1]):
        sol = radau5(ti, ti+dt/2, ysol, fcn_reac, 0, rtol=tol_reac, atol=tol_reac)
        ysol = sol.y
        #sol = rock4(ti, ti+dt, ysol, fcn_diff, rtol=tol_diff, atol=tol_diff)
        sol = radau5(ti, ti+dt, ysol, fcn_diff, 1, rtol=tol_diff, atol=tol_diff)
        ysol = sol.y
        sol = radau5(ti+dt/2, ti+dt, ysol, fcn_reac, 0, rtol=tol_reac, atol=tol_reac)
        ysol = sol.y

    return strang_result(ysol)

####################################################################
class parareal_result:
    def __init__(self, Y, T, nit):
        self.Y= Y
        self.T = T
        self.nit = nit

####################################################################
def parareal(tini, tend, nb_sub, yini, fcn, fcn_reac, fcn_diff, coarse_method, nc, tolc, fine_method, nf, tolf, max_iter, eps):

    if (max_iter <=0):
        print("max_iter must be greater than 0")
        return

    neq = len(yini)

    dT = (tend-tini)/nb_sub
    T = np.linspace(tini, tend, nb_sub+1)

    # Parareal solution, coarse and fine solution
    Y = np.zeros((neq, nb_sub+1))
    G = np.zeros((neq, nb_sub))
    F = np.zeros((neq, nb_sub))

    # initialization
    print("Initialization ...")
    Y[:,0] = yini
    for i, Ti in enumerate(T[:-1]):
        #print(i, Ti, Ti+dT, Y[:,i])
        sol_coarse = integrate(coarse_method, Ti, Ti+dT, nc, tolc, Y[:,i], fcn, fcn_reac, fcn_diff)
        #print(sol_coarse.y)
        G[:,i] = sol_coarse.y
        Y[:,i+1] = sol_coarse.y

    # compute fine approximation
    for i, Ti in enumerate(T[:-1]):
        sol_fine = integrate(fine_method, Ti, Ti+dT, nf, tolf, Y[:,i], fcn, fcn_reac, fcn_diff)
        F[:, i] = sol_fine.y

    # Parareal iteration
    for ipar in range(max_iter):
        print()
        print(f"Iteration {ipar+1}")

        Gm1 = np.copy(G)
        Ym1 = np.copy(Y)

        for i, Ti in enumerate(T[:-1]):
            sol_coarse = integrate(coarse_method, Ti, Ti+dT, nc, tolc, Y[:,i], fcn, fcn_reac, fcn_diff)
            G[:,i] = sol_coarse.y
            Y[:,i+1] = G[:,i] + (F[:,i] - Gm1[:,i])

        diff = np.linalg.norm(Y-Ym1)/np.sqrt(nb_sub*neq)
        print(f" || Yi - Yi-1 || = {diff:7.2e} (difference between two parareal iterations)")

        if (diff < eps) or (ipar == (max_iter-1)):
            break

        # compute fine approximation
        for i, Ti in enumerate(T[:-1]):
            sol_fine = integrate(fine_method, Ti, Ti+dT, nf, tolf, Y[:,i], fcn, fcn_reac, fcn_diff)
            F[:, i] = sol_fine.y

    return parareal_result(Y, T, ipar+1)

####################################################################
def compute_ref_sol(tini, tend, nb_sub, yini, fcn):

    T = np.linspace(tini, tend, nb_sub+1)
    Yref = np.zeros((len(yini),nb_sub+1))
    Yref[:,0] = yini
    for i, Ti in enumerate(T[:-1]):
        sol_ref = radau5(Ti, T[i+1], Yref[:,i], fcn, njac=1, atol=1.e-12, rtol=1.e-12)
        Yref[:,i+1] = sol_ref.y

    return parareal_result(Yref, T, 0)
