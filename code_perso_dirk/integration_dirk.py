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
from utilities import make_nd_array
from mylib import integration

# imports relatifs
#import sys, os
#sys.path.append(os.path.join('../ressources_utiles/TP_MAP551/notebook_pc_07', 'mylib'))
import rk_coeffs


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
    unm1 = np.copy(y0)
    solver = newton.newtonSolverObj()
    if initSol is None:
      Dres, LU, Dresinv = None, None, None
    else:
      Dres, LU, Dresinv = initSol.Dres, initSol.LU, initSol.Dresinv

    for it, tn in enumerate(t[:-1]):
        ## SUBSTEPS
#        print(f"t={t[it]:.16e}")
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

class strang_result:
    def __init__(self, t, y):
        self.t = t
        self.y = y

def strang(tini, tend, nt, yini, fcn_diff, fcn_reac, tol_diff=1.e-12, tol_reac=1.e-12,
           jac_reac=None, jac_diff=None, sparsity_pattern_reac=None, sparsity_pattern_diff=None,
           method_diff='Radau', method_reac='Radau'):
    t = np.linspace(tini, tend, nt)
    dt = (tend-tini)/(nt-1)
    ysol = yini[:]
    yout = np.zeros((yini.size,nt))
    yout[:,0] = yini[:]
    for it, ti in enumerate(t[:-1]):
        sol = scipy.integrate.solve_ivp(fun=fcn_reac, y0=ysol, t_span=[ti, ti+dt/2], method=method_reac, t_eval=[ti+dt/2],
                                        vectorized=False, rtol=tol_reac, atol=tol_reac, jac=jac_reac, jac_sparsity=sparsity_pattern_reac)
        ysol = sol.y[:,-1]
        sol = scipy.integrate.solve_ivp(fun=fcn_diff, y0=ysol, t_span=[ti, ti+dt], method=method_diff, t_eval=None, #[ti+dt],
                                        vectorized=False, rtol=tol_diff, atol=tol_diff, jac=jac_diff, jac_sparsity=sparsity_pattern_diff)
        ysol = sol.y[:,-1]
        
        sol = scipy.integrate.solve_ivp(fun=fcn_reac, y0=ysol, t_span=[ti+dt/2, ti+dt], method=method_reac, t_eval=[ti+dt],
                                        vectorized=False, rtol=tol_reac, atol=tol_reac, jac=jac_reac, jac_sparsity=sparsity_pattern_reac)
        ysol = sol.y[:,-1]
        yout[:,it+1] = ysol[:]
    return strang_result(t,yout)


if __name__=='__main__':
    #spring-mass test
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
    nt_dirk = 10
    nt_strang = 10
    T = np.array([0., 1.])

    #### compute reference solution
    t_start = pytime.time()
    solref = scipy.integrate.solve_ivp(fun=modelfun, y0=x_0, t_span=T, method='Radau', t_eval=None, vectorized=False, rtol=1e-12, atol=1e-12, jac=None)
    if solref.status!=0:
        raise Exception('ODE integration failed: {}'.format(solref.message))
    t_end = pytime.time()
    print('DIRK computed in {} s'.format(t_end-t_start))
    
    #### compute solution with DIRK
    #A,b,c = rk_coeffs.RK4coeffs()
    A,b,c,Ahat,bhat,chat = rk_coeffs.LDIRK343()
    t_start = pytime.time()
    sol_dirk = DIRK_integration(f=modelfun, y0=x_0, t_span=T, nt=nt_dirk, A=A, b=b, c=c, options=None, gradF=gradF,
                           bRosenbrockApprox=False, bUseCustomNewton=False)
    t_end = pytime.time()
    print('DIRK computed in {} s'.format(t_end-t_start))

    #### compute Strang solution
    t_start = pytime.time()
    sol_strang = strang(tini=T[0], tend=T[1], nt=nt_strang, yini=x_0,
#                        fcn_diff=modelfun, fcn_reac=lambda t,x:0*x,
                        fcn_reac=modelfun, fcn_diff=lambda t,x:0*x,
                       tol_diff=1.e-12, tol_reac=1.e-12)
    t_end = pytime.time()
    print('Strang computed in {} s'.format(t_end-t_start))


    ### PLOT SOLUTION
    fig,ax  = plt.subplots(2,1,sharex=True)
    for cc in [('ref', solref, None), ('DIRK', sol_dirk, '+'), ('Strang', sol_strang, 'x')]:
        sol = cc[1]; name = cc[0]; marker = cc[2]
        if sol.t.size>100:
            markevery = np.floor(sol.t.size/50).astype(int)
        else:
            markevery = 1
        ax[0].plot(sol.t, sol.y[0,:], label=name, marker=marker, markevery=markevery)
        ax[1].plot(sol.t, sol.y[1,:], label=name, marker=marker, markevery=markevery)
    ax[0].legend()
    ax[1].set_xlabel('t (s)')
    ax[1].set_ylabel('v (m/s)')
    ax[0].set_ylabel('x (m/s)')
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    fig.suptitle('Spring-mass system')

    #### PLOT ERROR TO REF
    fig,ax  = plt.subplots(2,1,sharex=True)
    import scipy.interpolate
    for cc in [('DIRK', sol_dirk), ('Strang', sol_strang)]:
        sol = cc[1]; name = cc[0]
        interped_ref = np.zeros_like(sol.y)
        interped_ref[0,:] = scipy.interpolate.interp1d(x=solref.t, y=solref.y[0,:], kind='linear')(sol.t)
        interped_ref[1,:] = scipy.interpolate.interp1d(x=solref.t, y=solref.y[1,:], kind='linear')(sol.t)
        error_x = np.abs( sol.y[0,:] - interped_ref[0,:] )
        error_v = np.abs( sol.y[1,:] - interped_ref[1,:] )
        ax[0].semilogy(sol.t, error_x, label=name)
        ax[1].semilogy(sol.t, error_v, label=name)
    ax[1].set_xlabel('t (s)')
    ax[1].set_ylabel('v (m/s)')
    ax[0].set_ylabel('x (m/s)')
    ax[0].legend()
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    fig.suptitle('Error compared to ref')
    