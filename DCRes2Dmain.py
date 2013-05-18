# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:43:51 2013

@author: ben
"""

import numpy as np
import scipy.sparse as ssp

# Set up a mesh
nx = 21.
ny = 11.

# def some functions to be used on each dimension

n2c = lambda(n): n[0:-1] + 0.5*np.diff(n)
e = lambda(n): np.ones([1, n-1])

def dnc(n):
    dnc = np.diff(n); 
    dnc = np.insert(dnc, 0, dnc[0]/2.);  # correct endpoints for a half step between
    dnc = np.append(dnc, dnc[-1]/2.)     # first/last cell center and the boundary
    return(dnc)
    
def neumann_diags(m,n):
    diags = np.vstack([-e(n)*1./m[1:], e(n)*1./m[:-1]])
    diags[1, 0] = diags[1, 0]*2; diags[0, -1] = diags[0, -1]*2
    return(diags)

# Create field and flux position vectors
x   = np.linspace(0,1,nx);
xc  = n2c(x)
dx  = np.diff(x)
dxc = dnc(xc)


y   = np.linspace(0,1,ny);
yc  = n2c(y)
dy  = np.diff(y)
dyc = dnc(yc)


# Unused so far
X  , Y  = np.meshgrid(x, y)
Xc , Yc = np.meshgrid(xc, yc)

Dc2nx = ssp.spdiags(neumann_diags(dxc,nx),[-1,0],nx,nx-1);
Dc2ny = ssp.spdiags(neumann_diags(dyc,ny),[-1,0],ny,ny-1);
Dn2cx = ssp.spdiags(np.vstack([-e(nx), e(nx)]),[0,1],nx-1,nx);
Dn2cy = ssp.spdiags(np.vstack([-e(ny), e(ny)]),[0,1],ny-1,ny);

Icy = ssp.eye(ny,ny-1)
Icx = ssp.eye(nx,nx-1)

Gcx = ssp.kron(Icx,Dc2ny)
Gcy = ssp.kron(Dc2nx,Icy)

Iny = ssp.eye(ny-1,ny)
Inx = ssp.eye(nx-1,nx)

Gnx = ssp.kron(Inx,Dn2cy)
Gny = ssp.kron(Dn2cx,Iny)


GRAD = ssp.vstack([Gcx, Gcy]);
DIV  = ssp.hstack([Gnx, Gny]);

# Some code to check that the GRAD, DIV operators work properly
#
Sigma = np.zeros(np.shape(Xc))
Sigma[5:7, 8:14] = 1.
Sigma = np.ravel(Sigma, order='F'); Sigma = Sigma.T

GRADsig = GRAD * Sigma
GRADsigx = np.reshape(GRADsig[0:nx*ny], np.shape(X), order='F')
GRADsigy = np.reshape(GRADsig[nx*ny-1:-1], np.shape(Y), order='F')

LAPsig = DIV * GRADsig
LAPsig = np.reshape(LAPsig, np.shape(Xc), order='F')