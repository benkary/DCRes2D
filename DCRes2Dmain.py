# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:43:51 2013

@author: ben
"""

import numpy as np
import numpy.linalg as nl
import scipy.sparse as ssp
import scipy.optimize as sop
import scipy.interpolate as si
import pylab as p

import ModOps as mo

#comments
# Set up a mesh
nx = 21.  ; nxc = nx - 1
ny = 11.  ; nyc = ny - 1

n2c = lambda(n): n[0:-1] + 0.5 * np.diff(n)
e   = lambda(n): np.ones([1, n])

## Create field and flux position vectors
x   = np.linspace(0,1,nx)
y   = np.linspace(0,1,ny)
dx  = np.diff(x)
dy  = np.diff(y)
    
## Create the GRAD and DIV operators to be used
GRAD = mo.getGRAD (x , y)
DIV  = mo.getDIV  (x , y)

# Some code to check that the GRAD, DIV operators work properly
longdim = ny*nxc + nyc*nx

Sigma = 1./40000* np.ones((nyc,nxc))
Sigma[7:10, 11:14] = 1./3000.
Sigma = np.ravel(Sigma, order='F'); Sigma = Sigma.T

GRADsig = GRAD * Sigma
GRADsigy = np.reshape(GRADsig[0 : nxc*ny], (ny, nxc), order='F')
GRADsigx = np.reshape(GRADsig[ny*nxc : longdim], (nyc, nx), order='F')

## Create the xy center to node averaging matrices apply them to Sigma and
#  diagonalize the results to interact with the DIV operator on the left
#  and the GRAD on the right
##

AVx, AVy = mo.getAV(nx, ny)

AVsigx = 1./(np.dot(AVx.todense(),1./Sigma)); 5
AVsigx = np.ravel(AVsigx, order='F')
AVsigy = 1./(np.dot(AVy.todense(),1./Sigma)); 
AVsigy = np.ravel(AVsigy, order='F')
diagAVsig = ssp.spdiags(np.append(AVsigx,AVsigy),0,longdim,longdim)

## Create a matrix to pad the matrix inversion and satisfy the condition e.Ty = 0

DX, DY = np.meshgrid(dx, dy)
h = DX * DY; h = np.ravel(h, order='F')
hmat = np.tile(h, (nxc*nyc, 1))

## Test the forward operation with a test source

q = np.zeros((nyc, nxc)); q[0,8] = 5; q[0,17] = -5; q = np.ravel(q, order='F')
V = nl.solve(DIV * diagAVsig * GRAD + hmat, q)

V = np.reshape(V,(nyc,nxc), order='F')
V_noisy = V + 0.05 * np.random.randn(nyc,nxc)





## Create the Jacobian for use in Gradient-based optimization routines


AVsigx = 1./(np.dot(AVx.todense(),1./Sigma)); 
AVsigx = np.ravel(AVsigx, order='F')
AVsigy = 1./(np.dot(AVy.todense(),1./Sigma)); 
AVsigy = np.ravel(AVsigy, order='F')
diagAVsig = ssp.spdiags(np.append(AVsigx,AVsigy),0,longdim,longdim)




#def fun():
#     
#sop.minimize()
#
      
## Need a projection operator that interpolates the solution to the forward
#  problem to the locations of the inversion input data - can maybe hold off on
#  this
##

xc  = mo.n2c(x)
yc  = mo.n2c(y)
Xc, Yc = meshgrid(xc,yc)
 

xdat = xc[13:20]
ydat = np.zeros((np.shape(xdat)))



  
#Vdat = Q * np.ravel(V, order='F')
#
## code to test the interpolation operator
#
#t  = np.array([0.1,0.2,0.3,0.4,0.5])
#ft = np.exp(t)*np.sin(pi*t)
#t_interp = np.array([0.12,0.21,0.28,0.35])
#
#tt1 = np.interp(t_interp,t,ft)
#Qtx, Qty = mo.get_Q(t_interp, t)
#tt2 = np.dot(Qt,ft)
#
#p.plot(t,ft)
#p.show()
#


## Goin for it

# Will want to re-implement these using the matrix vector formulation in ha0
# paper

# define an initial model
#m0 = 1./40000* np.ones((nyc,nxc))
#m0 = np.ravel(m0, order='F'); m0 = m0.T
#
#Beta = 1
#
#AVsigx = 1./(np.dot(AVx.todense(),1./m0)); 
#AVsigx = np.ravel(AVsigx, order='F')
#AVsigy = 1./(np.dot(AVy.todense(),1./m0)); 
#AVsigy = np.ravel(AVsigy, order='F')
#diagAVsig = ssp.spdiags(np.append(AVsigx,AVsigy),0,longdim,longdim)
#    
#A = DIV * diagAVsig * GRAD + hmat  
#
#G = (DIV * ssp.diags(GRAD * np.ravel(V, order='F'), 0) * diagAVsig**2 *
#      ssp.vstack((AVx, AVy)) * ssp.diags(m0, 0))
#
Qx, Qy = mo.get_Q(np.array([xdat, ydat]), np.array([xc, yc]))
#prV = Q * np.ravel(V_noisy, order='F')
#
#
#J = np.dot(-Q, np.dot(np.solve(A,G)))
#
#Rm  = DIV * GRAD * m0
#Rmm = DIV * GRAD
#
#Hred = J.T * J + Beta * Rmm
#p = Beta * Rm + J.T * (Q * np.solve(A,q) - Vnoisy)
#
#dm = np.solve()