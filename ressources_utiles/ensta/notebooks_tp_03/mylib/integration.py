import numpy as np
from scipy.integrate import solve_ivp


####################################################################
class ode_result:
    def __init__(self, y, t):
        self.y = y
        self.t = t

####################################################################
def integrate(method, tini, tend, nt, rtol, atol, yini, fcn):

    if method=='rk1':
        sol = rk1(tini, tend, nt, yini, fcn)
    elif method=='rk2':
        sol = rk2(tini, tend, nt, yini, fcn)
    elif method=='rk4':
        sol = rk4(tini, tend, nt, yini, fcn)
    elif method=='radau5':
        sol = solve_ivp(fcn, (tini, tend), yini, method="Radau", rtol=rtol, atol=atol)
    else:
        print("The integration method " + method + " is unknown")
        exit(1)
    return sol

####################################################################
def rk1(tini, tend, nt, yini, fcn):

    dt = (tend-tini) / (nt-1)
    t = np.linspace(tini, tend, nt)

    yini_array = np.array(yini)
    neq = yini_array.size

    y = np.zeros((neq, nt), order='FORTRAN')
    y[:,0] = yini_array

    for it, tn  in enumerate(t[:-1]):
        yn = y[:,it]
        y[:,it+1] = yn + dt*fcn(tn, yn)

    return ode_result(y, t)

####################################################################
def rk2(tini, tend, nt, yini, fcn):

    dt = (tend-tini) / (nt-1)
    t = np.linspace(tini, tend, nt)

    yini_array = np.array(yini)
    neq = yini_array.size

    y = np.zeros((neq, nt), order='FORTRAN')
    y[:,0] = yini_array

    for it, tn  in enumerate(t[:-1]):
        yn = y[:,it]
        k1 = fcn(tn, yn)
        k2 = fcn(tn + 0.5*dt, yn + dt*(0.5*k1))
        y[:,it+1] = yn + dt*k2

    return ode_result(y, t)

####################################################################
def rk3(tini, tend, nt, yini, fcn):

    dt = (tend-tini) / (nt-1)
    t = np.linspace(tini, tend, nt)

    yini_array = np.array(yini)
    neq = yini_array.size

    y = np.zeros((neq, nt), order='FORTRAN')
    y[:,0] = yini_array

    for it, tn  in enumerate(t[:-1]):
        yn = y[:,it]
        k1 = fcn(tn, yn)
        k2 = fcn(tn + 0.5*dt, yn + dt*(0.5*k1))
        k3 = fcn(tn + dt, yn + dt*(-k1 + 2*k2))
        y[:,it+1] = yn + (dt/6)*(k1+4*k2+k3)

    return ode_result(y, t)

####################################################################
def rk4(tini, tend, nt, yini, fcn):

    dt = (tend-tini) / (nt-1)
    t = np.linspace(tini, tend, nt)

    yini_array = np.array(yini)
    neq = yini_array.size

    y = np.zeros((neq, nt), order='FORTRAN')
    y[:,0] = yini_array

    for it, tn  in enumerate(t[:-1]):
        yn = y[:,it]
        k1 = fcn(tn, yn)
        k2 = fcn(tn + 0.5*dt, yn + dt*(0.5*k1))
        k3 = fcn(tn + 0.5*dt, yn + dt*(0.5*k2))
        k4 = fcn(tn + dt, yn + dt*k3)
        y[:,it+1] = yn + (dt/6)*(k1+2*k2+2*k3+k4)

    return ode_result(y, t)

####################################################################
class parareal_result:
    def __init__(self, Y, sol_fine, T, nit):
        self.Y= Y
        self.sol_fine = sol_fine
        self.T = T
        self.nit = nit

####################################################################
def parareal(tini, tend, nb_sub, yini, fcn, coarse_method, nc, rtolc, atolc, fine_method, nf, rtolf, atolf, max_iter, eps, scale_factor):

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

    diff = np.zeros(neq)

    # initialization
    print("Initialization ...")
    Y[:,0] = yini
    for i, Ti in enumerate(T[:-1]):
        #print(i, Ti, Ti+dT, Y[:,i])
        sol_coarse = integrate(coarse_method, Ti, Ti+dT, nc, rtolc, atolc, Y[:,i], fcn)
        #print(sol_coarse.y[:,-1])
        G[:,i] = sol_coarse.y[:,-1]
        Y[:,i+1] = sol_coarse.y[:,-1]

    # compute fine approximation
    for i, Ti in enumerate(T[:-1]):
        sol_fine = integrate(fine_method, Ti, Ti+dT, nf, rtolf, atolf, Y[:,i], fcn)
        F[:, i] = sol_fine.y[:,-1]

    # Parareal iteration
    for ipar in range(max_iter):
        print()
        print(f"Iteration {ipar+1}")

        Gm1 = np.copy(G)
        Ym1 = np.copy(Y)

        for i, Ti in enumerate(T[:-1]):
            sol_coarse = integrate(coarse_method, Ti, Ti+dT, nc, rtolc, atolc, Y[:,i], fcn)
            G[:,i] = sol_coarse.y[:,-1]
            Y[:,i+1] = G[:,i] + (F[:,i] - Gm1[:,i])

        for ieq in range(neq):
            diff[ieq] = np.linalg.norm((Y[ieq]-Ym1[ieq])/scale_factor[ieq])/np.sqrt(nb_sub)
            ##print(" scale factor : ", scale_factor[ieq])
            print(f" {ieq+1}th variable : || Yi - Yi-1 || = {diff[ieq]:7.2e} (difference between two parareal iterations)")

        if (np.max(diff) < eps) or (ipar == (max_iter-1)):
            sol_num = []
            for i, Ti in enumerate(T[:-1]):
                sol_num.append(integrate(fine_method, Ti, Ti+dT, nf, rtolf, atolf, Y[:,i], fcn))
            break

        # compute fine approximation
        for i, Ti in enumerate(T[:-1]):
            sol_fine = integrate(fine_method, Ti, Ti+dT, nf, rtolf, atolf, Y[:,i], fcn)
            F[:, i] = sol_fine.y[:,-1]

    return parareal_result(Y, sol_num, T, ipar+1)

####################################################################
def compute_ref_sol(tini, tend, nb_sub, yini, fcn, method):

    T = np.linspace(tini, tend, nb_sub+1)
    Yref = np.zeros((len(yini),nb_sub+1))
    Yref[:,0] = yini
    sol_ref = []
    for i, Ti in enumerate(T[:-1]):
        sol_ref.append(solve_ivp(fcn, (Ti, T[i+1]), Yref[:,i], method, rtol=1.e-12, atol=1.e-12))
        Yref[:,i+1] = sol_ref[i].y[:,-1]

    return parareal_result(Yref, sol_ref, T, 0)
