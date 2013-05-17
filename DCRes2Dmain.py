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

# Create field and flux position vectors
x   = np.linspace(0,1,nx);
xc  = x[0:nx-1] + 0.5*np.diff(x)
dx  = np.diff(x)
dxc = np.diff(xc); 
dxc = np.insert(dxc, 0, dxc[0]/2.);  # correct endpoints for a half step between
dxc = np.append(dxc, dxc[-1]/2.)     # first/last cell center and the boundary


y  = np.linspace(0,1,ny);
yc = y[0:ny-1] + 0.5*np.diff(y)
dy = np.diff(y)
dyc = np.diff(yc) 
dyc = np.insert(dyc, 0, dyc[0]/2.); 
dyc = np.append(dyc, dyc[-1]/2.)


# Unused so far
X  , Y  = np.meshgrid(x, y)
Xc , Yc = np.meshgrid(xc, yc)

ex  = np.ones([1, nx-1])
ey  = np.ones([1, ny-1])

diagsx = np.vstack([-ex*1./dxc[1:], ex*1./dxc[:-1]])
diagsx[1, 0] = diagsx[1, 0]*2; diagsx[0, -1] = diagsx[0, -1]*2

diagsy = np.vstack([-ey*1./dyc[1:], ey*1./dyc[:-1]])
diagsy[1, 0] = diagsy[1, 0]*2; diagsy[0, -1] = diagsy[0, -1]*2



Dc2nx = ssp.spdiags(diagsx,[-1,0],nx,nx-1); LookatDc2nx = Dc2nx.todense()
Dc2ny = ssp.spdiags(diagsy,[-1,0],ny,ny-1); LookatDc2ny = Dc2ny.todense()
Dn2cx = ssp.spdiags(np.vstack([-ex, ex]),[0,1],nx-1,nx); LookatDn2cx = Dn2cx.todense()
Dn2cy = ssp.spdiags(np.vstack([-ey, ey]),[0,1],ny-1,ny); LookatDn2cy = Dn2cy.todense()

Icy = ssp.eye(ny,ny-1)
Icx = ssp.eye(nx,nx-1)

Gcx = ssp.kron(Icx,Dc2ny)
Gcy = ssp.kron(Dc2nx,Icy)

Iny = ssp.eye(ny-1,ny)
Inx = ssp.eye(nx-1,nx)

Gnx = ssp.kron(Inx,Dn2cy)
Gny = ssp.kron(Dn2cx,Iny)


GRAD = ssp.vstack([Gcx, Gcy]); LookatGRAD = GRAD.todense()
DIV  = ssp.vstack([Gnx, Gny]); LookatDIV  = DIV.todense()

Sigma = np.zeros(np.shape(Xc))
Sigma[5:7, 8:14] = 1.
Sigma = np.ravel(Sigma, order='F'); Sigma = Sigma.T

GRADsig = GRAD * Sigma
GRADsigx = np.reshape(GRADsig[0:nx*ny], np.shape(X), order='F')
GRADsigy = np.reshape(GRADsig[nx*ny-1:-1], np.shape(Y), order='F')
