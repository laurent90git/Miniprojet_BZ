# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:32:09 2019
Etude des zones de stabilité et de précision des méthodes RK  

@author: Laurent
"""
import numpy as np
import matplotlib.pyplot as plt
import rk_coeffs

dpi = 80
figsize = (8,8)
def plotStabilityRegionRK_old(A,b,c, re_min=-5, re_max=10, im_min=-5, im_max=5, n_re=201, n_im=200 ):
    x = np.linspace(re_min, re_max, n_re)
    y = np.linspace(im_min, im_max, n_im)
    xx,yy = np.meshgrid(x,y)
    zz = xx+1j*yy
    
    s = np.size(b)
    e=np.ones((s,))
    R = lambda z: 1+z*np.dot(b, np.linalg.inv(np.eye(s)-z*A).dot(e))
    
    Rvec = np.vectorize(R)
    RR = Rvec(zz)
    rr = np.abs(RR)
    
    fig, ax = plt.subplots(1,1)
    # plot des axes Im et Re
    ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
    ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)
     
    # add contour of precision relative to exponential
    expp = np.exp(zz)
    pprecision = 100*np.abs(RR-expp)/np.abs(expp) # précision en pourcents
    map_levels = np.linspace(-5,5,50)
    cs            = ax.contourf(xx,yy, np.log10(pprecision), levels=map_levels, cmap = 'gist_earth')
    fig.colorbar(cs)
    
    levels = np.array([1,5,25,50,100,500]) #np.array(range(0,200,25))
    level_contour = ax.contour(xx,yy, pprecision, levels=levels, colors='k')
    ax.clabel(level_contour, inline=1, fontsize=10,  fmt='%1.0f')
    
    # add stability domain
    ax.contour(xx,yy,rr,levels=[0,1],colors='r')
    
    ## Axis description
    fig.suptitle('Domaine de stabilité (rouge), iso-contour de précision (%) et map de la précision (log10)')
    ax.set_xlabel(r'Re$(\lambda\Delta t)$')
    ax.set_ylabel(r'Im$(\lambda\Delta t)$')
    return fig#,xx,yy,rr




class testPrecision:
    def __init__(self, re_min, re_max, im_min, im_max, n_re, n_im, x0):
        """ This objects allows to numerically determine the precision of the chosen integrator for linear problems of the form y'= lambda*y
            This is done b scanning the eigenvalues lambda in a uniforme manner in the grid  [re_min, re_max] + i*[im_min, immax]
            with the initial value x0"""
        self.im = np.linspace(im_min, im_max, n_im)
        self.re = np.linspace(re_min, re_max, n_re)
        self.re_xx, self.im_yy = np.meshgrid(self.re, self.im)
        self.eigvals = self.re_xx + 1j*self.im_yy
        print('eigvals={}'.format(self.eigvals.shape))
        
        self.x0 = x0*np.ones_like(self.eigvals)
        
    def computeStabilityPolynomial(self, A,b,c):
        s = np.size(b)
        e=np.ones((s,))
        R = lambda z: 1+z*np.dot(b, np.linalg.inv(np.eye(s)-z*A).dot(e))
        
        bVectorize = False
        if bVectorize:
            Rvec = np.vectorize(R)
            RR = Rvec(self.eigvals)
        else:
            RR = np.zeros_like(self.eigvals)
            for i in range(self.eigvals.shape[0]):
                for j in range(self.eigvals.shape[1]):
                    try:
                        RR[i,j] = R(self.eigvals[i,j])
                    except Exception as e:
                        print(e)
                        print(f'i={i}, j={j}, lbda={self.eigvals[i,j]}')
                        raise e
        return RR
    
    def computeTheoreticalEvolution(self, nt):
        """ The solutions are of therm x0*exp(lambda*t) """
#        sol = np.hstack( [ (self.x0*np.exp(self.eigvals*(i+1))).ravel() for i in range(nt)] )
#        print('x0 = {}, sol={}'.format(self.x0.shape, sol.shape))
        sol = np.exp(self.eigvals*nt)*self.x0
        return sol
    
    def computeNthOrderExpoDL(n):
        """ Compute the Taylor expansion polynomial of the exponantial to the n-th order """
        pass #TODO
        
    def computeNumericalEvolution(self, nt):
        # 1 - mettre eigvals sous forme de vecteur
#        eigvals_vec = self.eigvals.reshape( (self.eigvals.size,) )
#        x0_vec = self.x0.reshape( (self.eigvals.size,) )
        RR = self.computeStabilityPolynomial(A,b,c)
#        sol = np.zeros( (x0_vec.size, nt) )
#        sol[:,0] = RR * x0_vec
#        for i in range(1,nt):
#            sol = RR * sol[:,i-1]
        sol = np.copy(self.x0)
        for i in range(nt):
            sol = RR*sol
        return sol
        
        
    def plotStabilityRegionRK(self, A,b,c,nt=1):
        xx,yy = self.re_xx, self.im_yy
        zz = self.eigvals
        x= self.re
        y= self.im
        
        RR   = self.computeNumericalEvolution(nt=nt)
        expp = self.computeTheoreticalEvolution(nt=nt) # self.x0*np.exp(zz*nt)

        rr = np.abs(RR/self.x0) # ratio d'augmentation
        
         
        # add contour of precision relative to exponential
        pprecision1 = 100*np.abs(RR-expp)/np.abs(expp) # précision en pourcents
        pprecision2 = 100*np.abs(RR-expp)/np.abs(RR) # précision en pourcents
        map_levels = np.linspace(-2,5,50)
        for pprecision,name in [(pprecision1, 'relative to theoretical exp'), (pprecision2, 'relative to num')]:
            ratio_height_over_width = np.abs( (np.max(y)-np.min(y))/(np.max(x)-np.min(x)) )
            fig, ax = plt.subplots(1,1,dpi=80, figsize=(8, 8*ratio_height_over_width))
            # plot des axes Im et Re
            ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
            ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)

            # contour de la précision relative
            cs = ax.contourf(xx,yy, np.log10(pprecision), levels=map_levels) #, cmap = 'gist_earth')
            fig.colorbar(cs)
            
            # add contour lines for precision
            # levels = np.round(np.logspace(0,2,5)).astype(int)
            levels = np.array( range(0,200,10) )
#            levels = np.array([1,5,25,50,100,500]) #np.array(range(0,200,25))
            level_contour = ax.contour(xx,yy, pprecision, levels=levels, colors='k')
            ax.clabel(level_contour, inline=1, fontsize=10,  fmt='%1.0f')
            
            # hachurer les zones > 100%
            error_sup_100 = np.where(pprecision >= 100.)
            temp = np.zeros_like(pprecision)
            temp[error_sup_100] = 1.
            cs   = ax.contourf(xx,yy, temp, colors=['w', 'w', 'w'], levels=[0,0.5,1.5],  #levels=[0., 1.0, 1.5],
                               hatches=[None,'//', '//'], alpha = 0.)

            # add stability domain
            ax.contour(xx,yy,rr,levels=[0,1],colors='r')
            
            ## Axis description
            fig.suptitle(f'Domaine de stabilité (rouge), iso-contour de précision (%)\n et map de la précision (log10) pour {nt} pas de temps erreur {name}')
            ax.set_xlabel(r'Re$(\lambda\Delta t)$')
            ax.set_ylabel(r'Im$(\lambda\Delta t)$')
#            ax.axis('equal')
        
        print(np.max(np.abs(expp)))
        print(np.max(np.abs(RR)))
        testnan = expp>1e10
        if np.any(testnan):
            print('issue in theoretical result')
            fig2, ax = plt.subplots(1,1)
            # plot des axes Im et Re
            ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
            ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)
            # plots overflow region
            map_levels = np.array([0.,0.5, 1.5])
            cs            = ax.contourf(xx,yy, 1.0*testnan, levels=map_levels)#, cmap = 'gist_earth')
            fig2.colorbar(cs)
            

        fig2, ax = plt.subplots(1,1)
        # plot des axes Im et Re
        ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
        ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)
        # plot "expansion" ratio
        map_levels = np.array(np.linspace(-3,3,20)) #np.logspace(-3,3,20)
        cs            = ax.contourf(xx,yy, np.log10(rr), levels=map_levels)#, cmap = 'gist_earth')
        fig2.colorbar(cs)
        ax.contour(xx,yy,rr,levels=[0,1],colors='r')    
        ax.set_title('expansion ratio of numerical solution')
        
        fig2, ax = plt.subplots(1,1)
        # plot des axes Im et Re
        ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
        ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)
        # plot "expansion" ratio
        map_levels = np.array(np.linspace(-3,3,20)) #np.logspace(-3,3,20)
        cs            = ax.contourf(xx,yy, np.log10(expp/self.x0), levels=map_levels)#, cmap = 'gist_earth')
        fig2.colorbar(cs)
        ax.contour(xx,yy,rr,levels=[0,1],colors='r')    
        ax.set_title('expansion ratio of numerical solution')
        return fig
    
if __name__=='__main__':
    ## test RK stability region
    N = 200
    A,b,c = rk_coeffs.getButcher('rk4')#RadauIIA-5')#'L-SDIRK-33') #SDIRK4()5L[1]SA-1')

#    plotStabilityRegionRK_old(A,b,c, re_min=-5, re_max=10, im_min=-5, im_max=5, n_re=N, n_im=N+2 )

#    test =             testPrecision(re_min=-10, re_max=10, im_min=-5, im_max=5, n_re=N, n_im=N+2, x0=1)
    test =             testPrecision(re_min=-20, re_max=20, im_min=-20, im_max=20, n_re=N, n_im=N+2, x0=1)
    test.plotStabilityRegionRK(A,b,c, nt=10)
    
    