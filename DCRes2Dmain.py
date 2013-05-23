# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:43:51 2013

@author: ben
"""

import numpy as np
import numpy.linalg as nl
import scipy.sparse as ssp
import ModOps as mo

# Set up a mesh
nx = 51.  ; nxc = nx - 1
ny = 31.  ; nyc = ny - 1

n2c = lambda(n): n[0:-1] + 0.5 * np.diff(n)
e   = lambda(n): np.ones([1, n])

## Create field and flux position vectors
x   = np.linspace(0,1,nx)
y   = np.linspace(0,1,ny)
dx  = np.diff(x)
dy  = np.diff(y)
    
#def dnc(n):
#    
#    dnc = np.diff(n); 
#    dnc = np.insert(dnc, 0, dnc[0]/2.);  # correct endpoints for a half step between
#    dnc = np.append(dnc, dnc[-1]/2.)     # first/last cell center and the boundary
#    
#    return(dnc)
#
#    
#def neumann_diags(m,n):
#    
#    d = np.vstack([-e(n-1)*1./m[1:], e(n-1)*1./m[:-1]])
#    d[1, 0] = d[1, 0]*2; d[0, -1] = d[0, -1]*2
#    
#    return(d)
#    
#def AV_diags(m):
#    
#    d = np.vstack([mo.e(m-1), mo.e(m-1)])
#    d[1, 0] = d[1, 0]*2; d[0, -1] = d[0, -1]*2
#    
#    return d
#    
#
#    
## Create field and flux position vectors
#x   = np.linspace(0,1,nx)
#xc  = n2c(x)
#dxc = dnc(xc)
#
#y   = np.linspace(0,1,ny)
#yc  = n2c(y)
#dyc = dnc(yc)
#
#Dc2nx = ssp.spdiags(neumann_diags(dxc,nx),[-1,0],nx,nxc)
#Dc2ny = ssp.spdiags(neumann_diags(dyc,ny),[-1,0],ny,nyc)
#
#Gcx = ssp.kron(ssp.eye(nxc,nxc), Dc2ny)
#Gcy = ssp.kron(Dc2nx, ssp.eye(nyc,nyc))
#
#GRAD = ssp.vstack([Gcx, Gcy])
#
#    

#
#Dn2cx = ssp.spdiags(np.vstack([-e(nxc), e(nxc)]),[0,1],nxc,nx)
#Dn2cy = ssp.spdiags(np.vstack([-e(nyc), e(nyc)]),[0,1],nyc,ny)
#
#
#Dnx = ssp.kron(ssp.eye(nxc,nxc), Dn2cy)
#Dny = ssp.kron(Dn2cx, ssp.eye(nyc,nyc))
#
#DIV  = ssp.hstack([Dnx, Dny])
#
#    


GRAD = mo.getGRAD (nx , ny)
DIV  = mo.getDIV  (nx , ny)

# Some code to check that the GRAD, DIV operators work properly
longdim = ny*nxc + nyc*nx

Sigma = 1./3000 * np.ones((nyc,nxc))
Sigma[13:18, 20:30] = 1./40000.
Sigma = np.ravel(Sigma, order='F'); Sigma = Sigma.T

GRADsig = GRAD * Sigma
GRADsigy = np.reshape(GRADsig[0 : nxc*ny], (ny, nxc), order='F')
GRADsigx = np.reshape(GRADsig[ny*nxc : longdim], (nyc, nx), order='F')

#LAPsig = np.dot(DIV, GRADsig)
#diagsx = AV_diags(nx)
#diagsy = AV_diags(ny)
#
#
#Ac2fx = 1./2 * ssp.spdiags(diagsx,[-1, 0],nx,nxc)
#Ac2fy = 1./2 * ssp.spdiags(diagsy,[-1, 0],ny,nyc)
#
##AVy = ssp.kron(ssp.eye(nx,nx), Ac2fy); AVy = AVy.todense()
##AVx = ssp.kron(Ac2fx, ssp.eye(ny-1,ny-1)); AVx = AVx.todense()
##AV  = np.dot(AVy, AVx)
#
#AVy = ssp.kron(ssp.eye(nxc,nxc), Ac2fy); #AVy = AVy.todense()
#AVx = ssp.kron(Ac2fx, ssp.eye(nyc,nyc)); #AVx = AVx.todense()
##AV  = ssp.vstack([AVx,AVy])
#
#
##AVsig = 1. / (np.dot(AV,(1./Sigma)))
##AVsig = np.reshape(AVsig, np.shape(X), order='F')

AVx, AVy = mo.getAV(nx, ny)


ll = 1./(np.dot(AVy.todense(),1./Sigma)); ll = np.ravel(ll, order='F')

#ll = np.reshape(ll, [ny, nxc], order='F')

mm = 1./(np.dot(AVx.todense(),1./Sigma)); mm = np.ravel(mm, order='F')

#mm = np.reshape(mm, [nyc, nx], order='F')

diagAVsig = ssp.spdiags(np.append(ll,mm),0,longdim,longdim)

DX, DY = np.meshgrid(dx, dy)
h = DX * DY; h = np.ravel(h, order='F')
hmat = np.tile(h, (nxc*nyc, 1))

#hmat = mo.get_hmat(dx,dy)


Aofu = np.dot(DIV, np.dot(diagAVsig, GRAD)) + hmat

q = np.zeros(nxc*nyc); q[15] = 5
V = nl.solve(Aofu,q)

V = np.reshape(V,(nyc,nxc), order='F')





