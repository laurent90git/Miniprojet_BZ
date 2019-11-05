#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:40:32 2019

@author: thomas
"""
import numpy as np
import copy
import collections
import math
pi=math.pi
import scipy.interpolate

def sigmoid_arctan(x, x0, width):
    n = 1
    temp = np.abs((x/width)**n)
    return (np.arctan(temp)*2/pi*np.sign(x-x0)+1)/2

#        plt.figure()
#        testx = np.linspace(-1,1,1000)
#        plt.plot(testx, sigmoid_arctan(testx,0,0.01), label='w=0.01')
#        plt.plot(testx, sigmoid_arctan(testx,0,0.1), label='w=0.1')
#        plt.plot(testx, sigmoid_arctan(testx,0,1.), label='w=1')
#        plt.legend()
#        plt.title('test sigmoid arctan')


def fastspy(A, ax, cmap='binary'):
    """"
    Parameters
    ----------
    A : coo matrix
    ax : axis
    """

    m, n = A.shape
    ax.hold(True)

    ax.imshow(A,interpolation='none',cmap=cmap)
    ax.colorbar()
    if 0:
        ax.scatter([i for i in range(np.size(A,1))],
                   [i for i in range(np.size(A,0))],
                   c=A.data, s=20, marker='s',
                   edgecolors='none', clip_on=False,
                   cmap=cmap)

    ax.axis('off')
    ax.axis('tight')
    ax.invert_yaxis()
    ax.hold(False)

def setupFiniteVolumeMesh(xfaces, meshoptions=None):
    """ Setup 1D spatial mash for finite volume, based on the positions of the faces of each cell """
    if meshoptions is None:
        meshoptions={}
    meshoptions['faceX'] = xfaces
    meshoptions['cellX'] = 0.5*(xfaces[1:]+xfaces[0:-1]) # center of each cell
    meshoptions['dxBetweenCellCenters'] = np.diff(meshoptions['cellX']) # gap between each consecutive cell-centers
    meshoptions['cellSize'] = np.diff(xfaces) # size of each cells
    assert not any(meshoptions['cellSize']==0.), 'some cells are of size 0...'
    assert not any(meshoptions['cellSize']<0.), 'some cells are of negative size...'
    assert not any(meshoptions['dxBetweenCellCenters']==0.), 'some cells have the same centers...'
    assert np.max(meshoptions['cellSize'])/np.min(meshoptions['cellSize']) < 1e10, 'cell sizes extrema are too different'
    # conveniency attributes for backward-compatibility  wtih finite-difference results post-processing
    meshoptions['x']  = meshoptions['cellX']
    meshoptions['dx'] = meshoptions['dxBetweenCellCenters']

    return meshoptions


def mergeDict(prioritary, other, level=0, genealogy=''):
    """ Recursively merge two dictionnaries, with precedance for the first one """
    if level==0:  #first call
        out =copy.deepcopy(prioritary)
    else:
        out = prioritary
    for key in other.keys():
        if key not in prioritary.keys():
            out[key] = other[key]
        else:
            #merge
            if isinstance(other[key], collections.Mapping):
                if isinstance(prioritary[key], collections.Mapping):
                    out[key] = mergeDict(prioritary[key], other[key], level=level+1, genealogy='{}.{}'.format(genealogy, key))
                else:
                    raise Exception('Priortary dict has key {}.{} of type {}, whereas it is of type {} in the other one'.format(genealogy, key, type(prioritary[key]), type(other[key])))
            else:
                if type(other[key]) == type(prioritary[key]):
                    pass #out[key] = prioritary[key]
                elif other[key]==None:
                    pass
                else:
                    raise Exception('Priortary dict has key {}.{} of type {}, whereas it is of type {} in the other one'.format(genealogy, key, type(prioritary[key]), type(other[key])))
    return out

def generateTimeVector(dictInputReference):
    """ Generates the time vector for integration """
    defaults = {'sCase': 'unsteady',
                'unsteady': {'dt':1e-6, 't_f':1e-3},
                'progressive':{
                        'dts': [1e-7,  1e-6, 1e-5], #successive time steps
                        'ntrelax': [ 100,  100], #number of transition time steps after the stabilization steps to transition from on dt to another
                        'nstab': [200,  100,   100], #number of time steps with fixed time steps for each separate dt provided
                        }}
    dictInput = mergeDict(prioritary=dictInputReference, other=defaults)
    if dictInput['sCase']=='unsteady':
            dt=dictInput['unsteady']['dt']
            time = np.arange(0.,dictInput['unsteady']['t_f'],dt)
    elif dictInput['sCase']=='unsteadyProgressif' or dictInput['sCase']=='progressive':
        dts =   dictInput['progressive']['dts'] #successive time steps
        ntrelax = dictInput['progressive']['ntrelax'] #number of transition time steps after the stabilization steps to transition from on dt to another
        nstab =  dictInput['progressive']['nstab'] #number of time steps with fixed time steps for each separate dt provided

        time = [0.]
        for i in range(len(dts)-1):
            for j in range(nstab[i]):
                time.append(time[-1] + dts[i])
            for j in range(ntrelax[i]):
                time.append(time[-1]+ (ntrelax[i]-j)/ntrelax[i]*dts[i] +  j/ntrelax[i]*dts[i+1])
        for j in range(nstab[-1]):
            time.append(time[-1] + dts[-1])
        time = np.array(time)
    else:
        raise Exception('unknown time stepping configuration "{}"'.format(dictInput['sCase']))

    if 'globalScaling' in dictInputReference.keys(): # global scaling to easily reduce time step sizes
        time = time*dictInputReference['globalScaling']
    return time

#raise Exception('attention  tu dois finir cette implémentation')
def interpExtrap1D(x, y, kind='linear'):
    """ Interpolateur qui extrapole avec des valeurs constantes, évite les problèmes posés par scipy.interp1d """
    import scipy.interpolate
#    from scipy import array
    def extrap1d(interpolator):
        xs = interpolator.x
        ys = interpolator.y

#        def pointwise(x):
#            if x < xs[0]:
#                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
#            elif x > xs[-1]:
#                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
#            else:
#                return interpolator(x)

        def ufunclike(xnew):
            Iinterp = np.intersect1d( np.where(xnew>xs[0])[0], np.where(xnew<xs[-1])[0] ).astype(int)
            Iextrap_low = np.where(xnew<=xs[0])
            Iextrap_up  = np.where(xnew>=xs[-1])
            ynew=np.zeros_like(xnew)
            ynew[Iinterp] = interpolator(xnew[Iinterp])
            ynew[Iextrap_low] = ys[0]
            ynew[Iextrap_up]  = ys[-1]
            return ynew
#            return np.array(map(pointwise, np.array(xnew)))

        return ufunclike
    return extrap1d(scipy.interpolate.interp1d(x,y,kind=kind))

if __name__=='__main__':
    import matplotlib.pyplot as plt
    sDict = {'sCase': 'unsteady',
            'unsteady': {'dt':1e-6, 't_f':1e-3},
            'progressive':{
                    'dts': [1e-7,  1e-6, 1e-5], #successive time steps
                    'ntrelax': [ 100,  100], #number of transition time steps after the stabilization steps to transition from on dt to another
                    'nstab': [200,  100,   100], #number of time steps with fixed time steps for each separate dt provided
                    }}
    time = generateTimeVector(sDict)
    plt.figure()
    plt.plot(time)
    plt.title('time vector')
    plt.xlabel('index')
    plt.ylabel('time')

    plt.figure()
    plt.plot(np.diff(time))
    plt.title('time vector gradient')
    plt.xlabel('index')
    plt.ylabel('dt')

    # test merge
    dict1 = {'a': 1.,
             'b':{'c':2,
                  'd':4,
                  }}
    dict2 = {'a': 1.3,
             'b':{'c':23,
                  'f': 5,
                  'e':{'test':'a word',}
                  }}

    dict3 = mergeDict(dict1, dict2)
    print(dict3)


    # test de l'interpolation avev extrapolation linéaire
    x= np.array([0., 1., 2.])
    y = np.array([0., 2., 4.])

    xnew = np.array([-1, 0., 0.5, 1.5, 2., 4.])
    ynew = interpExtrap1D(x,y,kind='linear')(xnew)
    ynew2 = scipy.interpolate.interp1d(x,y,kind='linear', fill_value='extrapolate')(xnew)
    plt.figure()
    plt.plot(x,y,label='original', marker='+', color='b')
    plt.scatter(xnew,ynew,  label='new custom', marker='o', color='r')
    plt.scatter(xnew,ynew2, label='interp1d', marker='x', color='g')
    plt.legend()
    plt.title('Validation de mon interpolation avec extrapolation constante')