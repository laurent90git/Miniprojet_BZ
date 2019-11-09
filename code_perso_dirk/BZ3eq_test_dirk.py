# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:48:12 2019

@author: Laurent
"""
from mylib import integration
from integration_dirk import rk_coeffs, DIRK_integration
import time as pytime
import scipy.integrate
import matplotlib.pyplot as plt
import newton
import numpy as np
from utilities import make_nd_array


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
        self.x = np.linspace(xmin,xmax,nx)
        
    def init_sol(self):
        """ Initialisatio "intelligente" de la solution """
        ylim = 0.05
        neq  = 3
        f = self.f
        q = self.q
        b = np.zeros(nx)
        c = np.zeros(nx)
        for inx in range( nx//20 +1 ):
            xcoor = 0.5
            ycoor = inx/(nx/20) - ylim        
            if (ycoor >= 0 and ycoor<=0.3*xcoor):
              b[inx] = 0.8
            else:
              b[inx] = q*(f+1)/(f-1)
        
            if ( ycoor>=0. ):
              c[inx] = q*(f+1)/(f-1) + np.arctan(ycoor/xcoor)/(8*np.pi*f)
            else:
              c[inx] = q*(f+1)/(f-1) + (np.arctan(ycoor/xcoor) + 2*np.pi)/(8*np.pi*f)

        b[nx//20+1:]=b[nx//20]
        c[nx//20+1:]=c[nx//20]
    
        a=(f*c)/(q+b)
        u = np.zeros(neq*nx)
        for inx in range(nx):
            irow = inx*neq
            u[irow]   = a[inx]
            u[irow+1] = b[inx]
            u[irow+2] = c[inx]
        assert (irow+2) == (neq*nx - 1)
        return u

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
#        for i in range(1,nx-1):
#            ydot[i,0] = Da*(a[i-1]-2*a[i]+a[i+1])*doverdxdx + 1/mu*( -q*a[i] - a[i]*b[i] + f*c[i] )
#            ydot[i,1] = Db*(b[i-1]-2*b[i]+b[i+1])*doverdxdx + 1/eps*( q*a[i] - a[i]*b[i] + b[i]*(1-b[i]) )
#            ydot[i,2] = Dc*(c[i-1]-2*c[i]+c[i+1])*doverdxdx + b[i] - c[i]
#        i = np.array(range(1,nx-1))
#        ydot[i,0] = Da*(a[i-1]-2*a[i]+a[i+1])*doverdxdx + 1/mu*( -q*a[i] - a[i]*b[i] + f*c[i] )
#        ydot[i,1] = Db*(b[i-1]-2*b[i]+b[i+1])*doverdxdx + 1/eps*( q*a[i] - a[i]*b[i] + b[i]*(1-b[i]) )
#        ydot[i,2] = Dc*(c[i-1]-2*c[i]+c[i+1])*doverdxdx + b[i] - c[i]
        
        ydot[1:nx-1,0] = Da*(a[0:nx-2]-2*a[1:nx-1]+a[2:nx])*doverdxdx + 1/mu*( -q*a[1:nx-1] - a[1:nx-1]*b[1:nx-1] + f*c[1:nx-1] )
        ydot[1:nx-1,1] = Db*(b[0:nx-2]-2*b[1:nx-1]+b[2:nx])*doverdxdx + 1/eps*( q*a[1:nx-1] - a[1:nx-1]*b[1:nx-1] + b[1:nx-1]*(1-b[1:nx-1]) )
        ydot[1:nx-1,2] = Dc*(c[0:nx-2]-2*c[1:nx-1]+c[2:nx])*doverdxdx + b[1:nx-1] - c[1:nx-1]
        
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
           
           
           
if __name__=='__main__':
    # model parameterization
    nx=101
    nt_dirk=5000
    nt_strang=100
    nvar = 3
    T = np.array([0., 10.])
    BZobj = BZ1D_3equations(eps=1e-2, mu=5e-5, f=3, q=2e-3,
             Da=2.5e-3, Db=2.5e-3, Dc = 1.5e-3,
             xmin=0., xmax=4., nx=nx)
    modelfun = BZobj.fcn
    gradF=None
    
    # initial conditions
    x_0 = BZobj.init_sol()
    fig,ax  = plt.subplots(nvar,1,sharex=True)
    markevery = 1
    for i in range(nvar):
        ax[i].plot(BZobj.x, x_0[i::3], marker='+', markevery=markevery)
    ax[-1].set_ylabel('x')
    fig.suptitle('initial solution')
    
    
    # compute reference solution
    t_start = pytime.time()
    if 1: # PYTHON version
        solref = scipy.integrate.solve_ivp(fun=modelfun, y0=x_0, t_span=T, method='Radau', t_eval=np.linspace(T[0], T[1], 1000),
                                           vectorized=False, rtol=1e-4, atol=1e-4, jac=gradF)
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
    #    A,b,c = rk_coeffs.RK4coeffs()
    A,b,c,Ahat,bhat,chat = rk_coeffs.LDIRK343()
#        A=np.array([[1,],]);  b=np.array([1]);  c=np.array([1])
    t_start = pytime.time()
    sol_dirk = DIRK_integration(f=modelfun, y0=x_0, t_span=T, nt=nt_dirk, A=A, b=b, c=c, options=None, gradF=gradF,
                           bRosenbrockApprox=False, bUseCustomNewton=True)
    ysol_dirk = [ sol_dirk.y[::3,:], sol_dirk.y[1::3,:], sol_dirk.y[2::3,:] ]
    t_end = pytime.time()
    print('DIRK solution computed in {} s with {} time steps'.format(t_end-t_start, sol_dirk.t.size))


    fig,ax  = plt.subplots(nvar,1,sharex=True)
    for i in range(nvar):
        for jj in range(0, solref.t.size, 50):
            ax[i].plot(BZobj.x, yref[i][:,jj], label='t={:.3e}'.format(solref.t[jj]))    
        ax[i].grid(which='both')
    ax[-1].set_ylabel('x')
    fig.suptitle('evolution of reference solution')
    
    
    fig,ax  = plt.subplots(nvar,1,sharex=True)
    markevery = 1
    for i in range(nvar):
        ax[i].plot(BZobj.x, ysol_dirk[i][:,-1], label='DIRK', marker='+', linestyle='', markevery=markevery, linewidth=2)
        ax[i].plot(BZobj.x, yref[i][:,-1], label='ref', marker=None, linewidth=0.5)
        ax[i].grid(which='both')
    ax[0].legend()
    ax[-1].set_ylabel('x')
    fig.suptitle('last time step BZ')
