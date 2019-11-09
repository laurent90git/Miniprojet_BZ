# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:49:22 2019

Modèle 1D différences finies de Burgers en 1D

@author: Laurent
"""

import time as pytime
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from utilities import make_nd_array
from mylib import integration
import rk_coeffs
import scipy.interpolate

class BurgersDiff1D():
    """ Equation de Burgers avec diffusion """
    def __init__(self, mu=1e-2, c=1., xmin=0., xmax=4., nx=100, osc_bc=lambda t: 0.) :
        """ osc_bs est un fonction du temps qui donne le flux gauche du/dx(0) """
        self.mu = mu
        self.c = c
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.osc_bc = osc_bc
        self.x = np.linspace(xmin,xmax,nx)
        self.dx = (xmax-xmin)/(nx+1)
        
    def init_sol(self):
        x_0 = np.ones_like(self.x)
        x_0[x_0.size//2:] = 0.
        return x_0

    def fcn(self, t, y):
#        mu  = self.mu
#        nx = self.nx
#        dx = self.dx
#        c  = self.c
#        doverdxdx = mu/(dx*dx)
#        ydot = np.zeros_like(y)
#        # neumann BC
#        flux_gauche = self.osc_bc(t)
#        y_gauche = y[0] - dx * flux_gauche
#        y_droit  = 0.
#        ydot[0]      = -c*(y[0]-y_gauche)/dx       + doverdxdx*(y_gauche-2*y[0]+y[1])
#        ydot[1:nx-1] = -c*(y[1:nx-1]-y[:nx-2])/dx  + doverdxdx*(y[0:nx-2] - 2*y[1:nx-1] + y[2:nx])
#        ydot[nx-1]   = -c*(y[nx-1]-y[nx-2])/dx     + doverdxdx*(y_droit-2*y[-1]+y[-2])
        ydot = self.fcn_conv(t,y) + self.fcn_diff(t,y)
        return ydot

    def fcn_diff(self, t, y):
        mu  = self.mu
        nx = self.nx
        dx = self.dx
        doverdxdx = mu/(dx*dx)
        ydot = np.zeros_like(y)
        # neumann BC
        flux_gauche = self.osc_bc(t)
        y_gauche = y[0] - dx * flux_gauche
        y_droit  = 0.
        ydot[0]      = doverdxdx*(y_gauche-2*y[0]+y[1])
        ydot[1:nx-1] = doverdxdx*(y[0:nx-2] - 2*y[1:nx-1] + y[2:nx])
        ydot[nx-1]   = doverdxdx*(y_droit-2*y[-1]+y[-2])
        return ydot

    def fcn_conv(self, t, y):
        nx = self.nx
        dx = self.dx
        c  = self.c
        ydot = np.zeros_like(y)
        # neumann BC
        flux_gauche = self.osc_bc(t)
        y_gauche = y[0] - dx * flux_gauche
        ydot[0]      = -c*(y[0]-y_gauche)/dx
        ydot[1:nx-1] = -c*(y[1:nx-1]-y[:nx-2])/dx
        ydot[nx-1]   = -c*(y[nx-1]-y[nx-2])/dx
        return ydot
    
    def fcn_radau(self, n, t, y, ydot, rpar, ipar):
        n_python = 3*self.nx
        t_python = make_nd_array(t, (1,))[0]
        y_python = make_nd_array(y, (n_python,))
        y_dot_python = self.fcn(t_python, y_python)
        for i in range(y_dot_python.size):
           ydot[i] = y_dot_python[i]


if __name__=='__main__':
    from integration_dirk import DIRK_integration, strang
    
    nx=200
    nt_dirk = 1000
    nt_strang = 2
    T = np.array([0., .3])
    u_conv = 0.2
    mu = 3e-2
    xmin, xmax = 0., 2.
    f_osc = 20 ##fréquence d'oscillation de la BC
    
    def osc_left_bc(t):
        """ fonction pour faire varier le flux gauche """
        return 10 + 0*40*np.cos(2*np.pi*t*f_osc)
    
    obj = BurgersDiff1D(mu=mu, c=u_conv, xmin=xmin, xmax=xmax, nx=nx, osc_bc=osc_left_bc)
    fcn = obj.fcn
    fcn_radau = obj.fcn_radau
    x_0 = obj.init_sol()
    gradF = None
    
    
    # compute jacobians to speed up things
    from scipy.sparse import diags
    import numpy as np
    k = np.array([np.ones(nx-1),-2*np.ones(nx),np.ones(nx-1)])
    offset = [-1,0,1]
    sparsity_pattern_diff = diags(k,offset).toarray()
    jac_diff = obj.mu/(obj.dx**2)*sparsity_pattern_diff
    
    k = np.array([-1*np.ones(nx-1), np.ones(nx)])
    offset = [-1,0]
    sparsity_pattern_reac = diags(k,offset).toarray()
    jac_reac = None #-obj.c/obj.dx * sparsity_pattern_reac
    
    sparsity_pattern = sparsity_pattern_diff
    
    # compute reference solution  
    t_start = pytime.time()
    if 1:
        solref = scipy.integrate.solve_ivp(fun=fcn, y0=x_0, t_span=T, method='RK45', t_eval=None, vectorized=False, rtol=1e-12, atol=1e-12, jac=None, jac_sparsity=sparsity_pattern)
    else:
        solref = integration.radau5(T[0], T[1], x_0, fcn_radau, njac=3, atol=1.e-12, rtol=1.e-12, iout=1)
    t_end = pytime.time()
    print('reference computed in {} s'.format(t_end-t_start))
    if solref.status!=0:
        raise Exception('ODE integration failed: {}'.format(solref.message))
    
    # compute solution with DIRK
    #    A,b,c = rk_coeffs.RK4coeffs()
    A,b,c,Ahat,bhat,chat = rk_coeffs.LDIRK343()
    t_start = pytime.time()
    sol=None
    sol_dirk = DIRK_integration(f=fcn, y0=x_0, t_span=T, nt=nt_dirk, A=A, b=b, c=c, options=None, gradF=gradF,
                           bRosenbrockApprox=False, bUseCustomNewton=True, initSol=sol)
    t_end = pytime.time()
    print('DIRK computed in {} s'.format(t_end-t_start))
    
    # Splitting
    t_start = pytime.time()
    sol_strang = strang(tini=T[0], tend=T[1], nt=nt_strang, yini=x_0, fcn_diff=obj.fcn_diff, fcn_reac=obj.fcn_conv,
                       tol_diff=1.e-12, tol_reac=1.e-12, jac_reac=jac_reac, jac_diff=jac_diff,
                       sparsity_pattern_reac=sparsity_pattern_reac, sparsity_pattern_diff=sparsity_pattern_diff,
                        method_diff='RK45', method_reac='RK45')

    t_end = pytime.time()
    print('Strang computed in {} s'.format(t_end-t_start))
    
    fig,ax  = plt.subplots(1,1,sharex=True)
    ax = [ax]
    markevery = 1
    ax[0].plot(obj.x, solref.y[:,-1], label='ref', marker=None)
    ax[0].plot(obj.x, sol_dirk.y[:,-1], label='DIRK', marker='+', linestyle='', markevery=markevery)
    ax[0].plot(obj.x, sol_strang.y[:,-1], label='Strang', marker='x', linestyle='', markevery=markevery)
    ax[0].legend()
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('u')
    ax[0].grid(which='both')
    fig.suptitle('Burgers with viscosity={:.3e}'.format(obj.mu))

    fig,ax  = plt.subplots(1,1,sharex=True)
    ax = [ax]
    for cc in [('DIRK', sol_dirk, '+'), ('Strang', sol_strang, 'x')]:
        sol = cc[1]; name = cc[0]; marker=cc[2]
        error = np.abs( sol.y[:,-1] - solref.y[:,-1] )
        ax[0].semilogy( obj.x, error, label='{} err={}'.format(name, np.linalg.norm(error)), marker=marker )
    ax[0].set_xlabel('x')
    ax[0].legend()
    ax[0].grid(which='both')
    fig.suptitle('Error compared to ref')
    plt.show()
    
    
    # convergence rate of splitting methods with non-homogeneous constant BCs
    if 1:   
        nt_vec = 1.+np.power( 2., range(0,8))  # Strang
        sol_name = 'strang'
    else:
        nt_vec = 1.+np.power( 2., range(7,12)) # DIRK
        sol_name = 'DIRK'
    nt_vec = np.floor(nt_vec).astype(int)
    f_osc_vec = np.linspace(0,100,4) #fréquence d'oscillation de la BC
    sols = [[None for i in nt_vec] for j in f_osc_vec]
    solrefs = [None for j in f_osc_vec]

    for j,f_osc in enumerate(f_osc_vec):
        def osc_left_bc(t):
            """ fonction pour faire varier le flux gauche """
            return  5 + 40*np.cos(2*np.pi*t*f_osc)
        
        obj = BurgersDiff1D(mu=mu, c=u_conv, xmin=xmin, xmax=xmax, nx=nx, osc_bc=osc_left_bc)
        err_2   = 0.*np.zeros(nt_vec.size)
        err_1   = 0.*np.zeros(nt_vec.size)
        err_inf = 0.*np.zeros(nt_vec.size)

        # compute ref sol
        solrefs[j] = scipy.integrate.solve_ivp(fun=obj.fcn, y0=x_0, t_span=T, method='Radau', t_eval=None, vectorized=False, rtol=1e-12, atol=1e-12, jac=None, jac_sparsity=sparsity_pattern)

        # compute different strang solution
        for i,nt in enumerate(nt_vec):
            print(' | {}'.format(nt), end='')
            if sol_name=='DIRK':
                sols[j][i] = DIRK_integration(f=obj.fcn, y0=np.copy(x_0), t_span=T, nt=nt, A=A, b=b, c=c, options=None, gradF=gradF,
                           bRosenbrockApprox=False, bUseCustomNewton=True, initSol=None)
            elif sol_name=='strang':
                sols[j][i] = strang(tini=T[0], tend=T[1], nt=nt, yini=x_0, fcn_diff=obj.fcn_diff, fcn_reac=obj.fcn_conv,
                                tol_diff=1.e-12, tol_reac=1.e-12,# jac_reac=jac_reac, jac_diff=jac_diff,
                                sparsity_pattern_reac=sparsity_pattern_reac, sparsity_pattern_diff=sparsity_pattern_diff,
                                method_diff='RK45', method_reac='RK45')
            else:
                raise Exception('unknown integrator {}'.format(sol_name))
        for i,nt in enumerate(nt_vec):    
            err = sols[j][i].y[:,-1] - solrefs[j].y[:,-1]
            err_2[i] = np.linalg.norm( err, ord=None )
            err_1[i] = np.linalg.norm( err, ord=1 )
            err_inf[i] = np.linalg.norm( err, ord=np.inf )
            
        err_mat = np.vstack((err_inf, err_1, err_2))
        dt_split_vec = (T[1]-T[0])/nt_vec
        
        fig,ax = plt.subplots(err_mat.shape[0],1,sharex=True)
    #    names = [r'$||\eps||$_{\inf}', r'$||\eps||$_{1}', r'$||\eps||$_{2}']
        names = ['inf','1','2']
        for i in range(err_mat.shape[0]):
            name = names[i]
            ax[i].loglog(dt_split_vec, err_mat[i,:], marker='+', label='num')
            ax[i].loglog(dt_split_vec, err_mat[i,0]*(nt_vec[0]/nt_vec)**2, label='order 2')
            ax[i].grid(which='both')
            ax[i].legend()
            ax[i].set_title(name)
        fig.suptitle('Norms of {} error for f={}'.format(sol_name, f_osc))
        plt.show()
        
    