# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:24:00 2019

@author: Laurent
"""
import time as pytime
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import newton
import ctypes

# imports relatifs
import sys, os
#sys.path.append(os.path.join('../ressources_utiles/TP_MAP551/notebook_pc_07', 'mylib'))
from mylib import integration
import rk_coeffs


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

class BZ1D_3equations():
    """ Modèle BZ à 3 équations, paramétrable """
    def __init__(self, eps=1e-2, mu=5e-5, f=3, q=2e-3,
                 Da=2.5e-3, Db=2.5e-3, Dc = 1.5e-3,
                 xmin=0., xmax=4., nx=100) :
        self.mu = mu
        self.eps = eps
        self.f = f
        self.q = q
        self.Da = Da
        self.Db = Db
        self.Dc = Dc
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.dx = (xmax-xmin)/(nx+1)

    def fcn(self, t, y):
        """ y est de la forme [ a0, b0, c0, a1, b1 , c1 ] pour assurer une
        structure tridiagonale par bloc pour la Jacobienne """
        mu  = self.mu
        eps = self.eps
        f   = self.f
        q = self.q
        Da = self.Da
        Db = self.Db
        Dc = self.Dc
        nx = self.nx
        dx = self.dx
        doverdxdx = 1/(dx*dx)

        ymat = np.reshape(y, (3,nx), order='F').T
        ydot = np.zeros(ymat.shape)

        a = ymat[:,0]
        b = ymat[:,1]
        c = ymat[:,2]

        i=0
        # neumann BC
        ydot[i,0] = Da*(-2*a[i]+a[i+1])*doverdxdx + 1/mu*( -q*a[i] - a[i]*b[i] + f*c[i] )
        ydot[i,1] = Db*(-2*b[i]+b[i+1])*doverdxdx + 1/eps*( q*a[i] - a[i]*b[i] + b[i]*(1-b[i]) )
        ydot[i,2] = Dc*(-2*c[i]+c[i+1])*doverdxdx + b[i] - c[i]
        for i in range(1,nx-1):
            ydot[i,0] = Da*(a[i-1]-2*a[i]+a[i+1])*doverdxdx + 1/mu*( -q*a[i] - a[i]*b[i] + f*c[i] )
            ydot[i,1] = Db*(b[i-1]-2*b[i]+b[i+1])*doverdxdx + 1/eps*( q*a[i] - a[i]*b[i] + b[i]*(1-b[i]) )
            ydot[i,2] = Dc*(c[i-1]-2*c[i]+c[i+1])*doverdxdx + b[i] - c[i]
        i=nx-1
        ydot[i,0] = Da*(a[i-1]-2*a[i])*doverdxdx + 1/mu*( -q*a[i] - a[i]*b[i] + f*c[i] )
        ydot[i,1] = Db*(b[i-1]-2*b[i])*doverdxdx + 1/eps*( q*a[i] - a[i]*b[i] + b[i]*(1-b[i]) )
        ydot[i,2] = Dc*(c[i-1]-2*c[i])*doverdxdx + b[i] - c[i]

        return ydot.ravel(order='C')

    def fcn_radau(self, n, t, y, ydot, rpar, ipar):
        n_python = 3*self.nx
        t_python = make_nd_array(t, (1,))[0]
        y_python = make_nd_array(y, (n_python,))
        y_dot_python = self.fcn(t_python, y_python)
        for i in range(y_dot_python.size):
           ydot[i] = y_dot_python[i]

def computeJacobian(modelfun,x, options, bUseComplexStep):
    """
    Method to numerically compute the Jacobian of the function modelfun with
    respect to all the components of the input vector x.

    INPUTS:
        - modelfun:
            a function of the type y=modelfun(x)
        - x: (numpy 1D-array)
            the vector around which the Jacobian must be computed
        - Options is a dictionnary including 2 fields:
            - bUseComplexStep: (boolean)
                if True, a complex pertubation is used, allowing for machine-precision-level
                accuracy in the determinaiton fo the Jacobian
                if False, finite differences are used with adaptive pertubation size
            - bVectorisedModelFun: (boolean)
                True if the modelfun can accept a vectorised input such as a matrix
                [x, x1, x2, x3] instead of just the vector x. This way, the modelfun can
                be called less often and more efficiently

    OUTPUT:
        - Jac, the Jacobian of the function
    """
    n_x = np.size(x)
    hcpx = 1e-50
    # multiple perturbed calls
    if bUseComplexStep:
        res = modelfun(x)
        Dres = np.zeros( (np.size(res,0), np.size(x,0)))
        for ip in range(n_x):
            perturbation = np.zeros(np.size(x), dtype='cfloat')
            perturbation[ip] = hcpx*1j
            perturbation = x + perturbation
            resP = modelfun(perturbation)
            Dres[:,ip] = np.imag(resP)/hcpx
        #res = real(resP)
    else:
        res = modelfun(x)
        Dres = np.zeros( (np.size(res,0), np.size(x,0)) )
        current_h = np.zeros((n_x,1))
        for ip in range(n_x):
            current_h[ip] = np.max([1e-5*abs(x[ip]), 1e-5]) # perturbation's size

            perturbation = np.zeros(np.size(x))
            perturbation[ip] = current_h[ip]

            perturbation = x + perturbation
            resP = modelfun(perturbation)
            Dres[:,ip] = (resP-res)/current_h[ip]
    return Dres

def computeJacobianTriDiag(modelfun,x, options, bUseComplexStep):
    """
    Method to numerically compute the Jacobian of the function modelfun with
    respect to all the components of the input vector x.

    INPUTS:
        - modelfun:
            a function of the type y=modelfun(x)
        - x: (numpy 1D-array)
            the vector around which the Jacobian must be computed
        - Options is a dictionnary including 2 fields:
            - bUseComplexStep: (boolean)
                if True, a complex pertubation is used, allowing for machine-precision-level
                accuracy in the determinaiton fo the Jacobian
                if False, finite differences are used with adaptive pertubation size
            - bVectorisedModelFun: (boolean)
                True if the modelfun can accept a vectorised input such as a matrix
                [x, x1, x2, x3] instead of just the vector x. This way, the modelfun can
                be called less often and more efficiently

    OUTPUT:
        - Jac, the Jacobian of the function
    """
    n_x = np.size(x)
    hcpx = 1e-50
    # multiple perturbed calls
    if bUseComplexStep:
        res = modelfun(x)
        Dres = np.zeros( (np.size(res,0), np.size(x,0)))
        for ip in range(n_x):
            perturbation = np.zeros(np.size(x), dtype='cfloat')
            perturbation[ip] = hcpx*1j
            perturbation = x + perturbation
            resP = modelfun(perturbation)
            Dres[:,ip] = np.imag(resP)/hcpx
        #res = real(resP)
    else:
        res = modelfun(x)
        Dres = np.zeros( (np.size(res,0), np.size(x,0)) )
        current_h = np.zeros((n_x,1))
        for ip in range(n_x):
            current_h[ip] = np.max([1e-5*abs(x[ip]), 1e-5]) # perturbation's size

            perturbation = np.zeros(np.size(x))
            perturbation[ip] = current_h[ip]

            perturbation = x + perturbation
            resP = modelfun(perturbation)
            Dres[:,ip] = (resP-res)/current_h[ip]
    return Dres

def DIRK_integration(f, y0, t_span, nt, A, b, c, options, gradF=None,
                     bRosenbrockApprox = False, bUseCustomNewton = False, initSol=None):
    """ Performs the integration of the system dy/dt = f(t,y)
        from t=t_span[0] to t_span[1], with initial condition y(t_span[0])=y0.
        The RK method described by A,b,c may be explicit or diagonally-implicit.
        - f      :  (function handle) model function (time derivative of y)
        - y0     :  (1D-array)        initial condition
        - t_span :  (1D-array)        array of 2 values (start and end times)
        - nt     :  (integer)         number of time steps
        - A      :  (2D-array)        Butcher table of the chosen RK method
        - b      :  (1D-array)        weightings for the quadrature formula of the RK methods
        - c      :  (1D-array)        RK substeps time
        - options:  (dict)            used to parametrize the newton method and the jacobian determination

        - gradF  :  (function handle, optional) function returning a 2D-array (Jacobian df/dy)
        - bRosenbrockApprox: (boolean) use only one simplified newton iteration
        - bUseCustomNewton: use a custom newton solver (slightly quicker than scipy's one ?)
        - initSol : (structure) contains useful info for warm stars (ie already computed LU factorization of the Jacobian...)
        """
    out = scipy.integrate._ivp.ivp.OdeResult()
    t = np.linspace(t_span[0], t_span[1], nt)

    nx = np.size(y0)
    y = np.zeros(( nx, nt ))
    y[:,0] = y0
    dt = (t_span[1]-t_span[0]) / (nt-1)

    s = np.size(b)
     # to perform approximate bu faster resolution of the substeps
     # use custm newton solver instead of scipy method

    if gradF==None: # if the user does not provide a custom function to provide the Jacobian of f (for example with an analytical formulation)
        gradF = lambda t,x: computeJacobian(modelfun=lambda x: f(t,x),
                                            x=x,
                                            options=options,
                                            bUseComplexStep=False)

    K= np.zeros((np.size(y0), s))
    unm1 = y0[:]
    solver = newton.newtonSolverObj()
    if initSol is None:
      Dres, LU, Dresinv = None, None, None
    else:
      Dres, LU, Dresinv = initSol.Dres, initSol.LU, initSol.Dresinv

    for it, tn in enumerate(t[:-1]):
        ## SUBSTEPS
#        unm1 = y[:,it]
        for isub in range(s): # go through each substep
            temp = np.zeros(np.shape(y0))
            for j in range(isub):
                temp    = temp  +  A[isub,j] * K[:,j]
            vi = unm1 + dt*( temp )

            if A[isub,isub]==0: # explicit step
                kni = f(tn+c[isub]*dt, vi)
            else:
                if bRosenbrockApprox: # assume f is very close to being an affine function of x
                    if isub==0:#evaluate jacobian
                        rosenGradF = gradF(tn+c[isub]*dt, unm1)
                    temp = f(tn+c[isub]*dt, vi)
                    kni = np.linalg.solve(a = np.eye(nx)-dt*A[isub,isub]*rosenGradF, #gradF(tn+c[isub]*dt, vi),
                                          b = temp)
                else: # solve the complete non-linear system via a Newton method
    #                gradFun = None
                    tempfun = lambda kni: kni - f(tn+c[isub]*dt, vi + dt*A[isub,isub]*kni)
                    gradFun = lambda kni: 1 - dt*A[isub,isub]*gradF(tn+c[isub]*dt, vi + dt*A[isub,isub]*kni)
                    if bUseCustomNewton:
                        kni, Dres, LU, Dresinv = solver.solveNewton(fun=tempfun,
                                                    x0=K[:,0],
                                                    initJac=Dres,
                                                    initLU=LU,
                                                    initInv=Dresinv,
                                                    options={'eps':1e-8, 'bJustOutputJacobian':False, 'nIterMax':50, 'bVectorisedModelFun':False,
                                                             'bUseComplexStep':False, 'bUseLUdecomposition':True, 'bUseInvertJacobian':False,
                                                             'bModifiedNewton':True, 'bDampedNewton':False, 'limitSolution':None, 'bDebug':False,
                                                             'bDebugPlots':False, 'nMaxBadIters':2, 'nMaxJacRecomputePerTimeStep':5} )

                    else:
                        kni = scipy.optimize.fsolve(func= tempfun,
                                              x0=K[:,0],
                                              fprime=None,
                                              band=(5,5), #gradFun
                                              epsfcn = 1e-7,
                                              args=(),)
            K[:,isub] = kni #f(tn+c[isub]*dt, ui[:,isub])
        ## END OF STEP --> reaffect unm1
        for j in range(s):
            unm1[:] = unm1[:] + dt*b[j]*K[:,j]
        y[:,it+1] = unm1[:]
    # END OF INTEGRATION
    out.y = y[:,:nt+1]
    out.t = t
    if bUseCustomNewton:
      out.Dres = Dres
      out.Dresinv = Dresinv
      out.LU = LU
    return out


if __name__=='__main__':

    if 0: #spring-mass
                print('testing DIRK with a simple spring-mass model')
                options = {'k_sur_m': 33.,
                           'bDebug': True,
                           'bVectorisedModelFun': False,
                           'bUseComplexStep': False,
                           }
                modelA = np.array( ( (0,1),(-options['k_sur_m'], 0) ))
                def modelfun(t,x,options={}):
                    Xdot = np.dot(modelA, x)
                    return Xdot
                gradF = lambda t,x: modelA
                x_0 = np.array((0.3,0.))
                nt = 1000
                T = np.array([0., 1.])

                # compute reference solution
                solref = scipy.integrate.solve_ivp(fun=modelfun, y0=x_0, t_span=T, method='Radau', t_eval=None, vectorized=False, rtol=1e-6, atol=1e-6, jac=None)
                if solref.status!=0:
                    raise Exception('ODE integration failed: {}'.format(solref.message))

                # compute solution with DIRK
                import rk_coeffs
            #    A,b,c = rk_coeffs.RK4coeffs()
                A,b,c,Ahat,bhat,chat = rk_coeffs.LDIRK343()

            #    A=np.array([[1,],])
            #    b=np.array([1])
            #    c=np.array([1])
                import time as pytime
                t_start = pytime.time()
                sol=None
                sol = DIRK_integration(f=modelfun, y0=x_0, t_span=T, nt=nt, A=A, b=b, c=c, options=None, gradF=gradF,
                                       bRosenbrockApprox=False, bUseCustomNewton=False, initSol=sol)
                t_end = pytime.time()
                print('computed in {} s'.format(t_end-t_start))
            #    sol = solref

                fig,ax  = plt.subplots(2,1,sharex=True)
                if sol.t.size>100:
                    markevery = np.floor(sol.t.size/50).astype(int)
                else:
                    markevery = 1
                ax[0].plot(sol.t, sol.y[0,:], label='DIRK', marker='+', linestyle='', markevery=markevery)
                ax[0].plot(solref.t, solref.y[0,:], label='ref', marker=None)

                ax[1].plot(sol.t, sol.y[1,:], label='DIRK', marker='+', linestyle='', markevery=markevery)
                ax[1].plot(solref.t, solref.y[1,:], label='ref', marker=None)

                ax[0].legend()
                ax[1].set_xlabel('t (s)')
                ax[1].set_ylabel('v (m/s)')
                ax[0].set_ylabel('x (m/s)')

                ax[0].grid(which='both')
                ax[1].grid(which='both')
                fig.suptitle('Spring-mass system')

                fig,ax  = plt.subplots(2,1,sharex=True)
                import scipy.interpolate
                interped_ref = np.zeros_like(sol.y)
                interped_ref[0,:] = scipy.interpolate.interp1d(x=solref.t, y=solref.y[0,:], kind='linear')(sol.t)
                interped_ref[1,:] = scipy.interpolate.interp1d(x=solref.t, y=solref.y[1,:], kind='linear')(sol.t)
                error_x = np.abs( sol.y[0,:] - interped_ref[0,:] )
                error_v = np.abs( sol.y[1,:] - interped_ref[1,:] )
                ax[0].semilogy(sol.t, error_x)
                ax[1].semilogy(sol.t, error_v)

                ax[1].set_xlabel('t (s)')
                ax[1].set_ylabel('v (m/s)')
                ax[0].set_ylabel('x (m/s)')

                ax[0].grid(which='both')
                ax[1].grid(which='both')
                fig.suptitle('Error compared to ref')
    else:
        # model parameterization
        nx=100
        nt=1000
        T = np.array([0., 1.])
        BZobj = BZ1D_3equations(eps=1e-2, mu=5e-5, f=3, q=2e-3,
                 Da=2.5e-3, Db=2.5e-3, Dc = 1.5e-3,
                 xmin=0., xmax=4., nx=nx)
        modelfun = BZobj.fcn
        gradF=None

        # initial conditions
        a0  = 1+0*np.linspace(0,1,nx)
        b0  = 1+0*np.linspace(0,1,nx)
        c0  = 1+0*np.linspace(0,1,nx)
        x_0 = np.vstack((a0,b0,c0)).ravel(order='F')
        mesh_x = np.linspace(0,4,nx)


        # compute reference solution
        t_start = pytime.time()
        if 0: # PYTHON version
            solref = scipy.integrate.solve_ivp(fun=modelfun, y0=x_0, t_span=T, method='Radau', t_eval=None, vectorized=False, rtol=1e-12, atol=1e-12, jac=gradF)
            if solref.status!=0:
                raise Exception('ODE integration failed: {}'.format(solref.message))
            yref = [ solref.y[::3,:], solref.y[1::3,:], solref.y[2::3,:] ]
            solref.nstep = solref.t.size
        else: # FORTRAN VERSION
          solref = integration.radau5(T[0], T[1], x_0, BZobj.fcn_radau, njac=1, atol=1.e-12, rtol=1.e-12)
          solref.y = solref.y[:,np.newaxis]
          yref = [ solref.y[::3,:], solref.y[1::3,:], solref.y[2::3,:] ]
        t_end = pytime.time()
        print('Reference solution computed in {} s with {} time steps'.format(t_end-t_start, solref.nstep))

        # compute solution with DIRK
        if 1:
            #    A,b,c = rk_coeffs.RK4coeffs()
            #    A,b,c,Ahat,bhat,chat = rk_coeffs.LDIRK343()
                A=np.array([[1,],]);  b=np.array([1]);  c=np.array([1])
                t_start = pytime.time()
                sol=None
                sol = DIRK_integration(f=modelfun, y0=x_0, t_span=T, nt=nt, A=A, b=b, c=c, options=None, gradF=gradF,
                                       bRosenbrockApprox=False, bUseCustomNewton=True, initSol=sol)
                ysol = [ sol.y[::3,:], sol.y[1::3,:], sol.y[2::3,:] ]
                t_end = pytime.time()
                print('DIRK solution computed in {} s with {} time steps'.format(t_end-t_start, sol.t.size))
        else:
          ysol=yref
          sol=solref


        nvar = 3
        fig,ax  = plt.subplots(nvar,1,sharex=True)
        if sol.t.size>2e3:
            markevery = np.floor(sol.t.size/50).astype(int)
        else:
            markevery = 1
        for i in range(nvar):
            ax[i].plot(mesh_x, ysol[i][:,-1], label='DIRK', marker='+', linestyle='', markevery=markevery, linewidth=2)
            ax[i].plot(mesh_x, yref[i][:,-1], label='ref', marker=None, linewidth=0.5)
            ax[i].grid(which='both')
        ax[0].legend()
        ax[-1].set_ylabel('x (m/s)')
        fig.suptitle('last time step BZ')
