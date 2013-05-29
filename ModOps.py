# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:21:44 2013

@author: ben
"""
import scipy.sparse as ssp
import numpy as np

#:------------------------------------------------------------------:#

n2c = lambda(n): n[0:-1] + 0.5*np.diff(n)
e   = lambda(n): np.ones([1, n])

def dnc(n):
    
    dnc = np.diff(n); 
    dnc = np.insert(dnc, 0, dnc[0]/2.);  # correct endpoints for a half step between
    dnc = np.append(dnc, dnc[-1]/2.)     # first/last cell center and the boundary
    
    return(dnc)

    
def neumann_diags(m, n):
    
    d = np.vstack([-e(n-1)*1./m[1:], e(n-1)*1./m[:-1]])
    d[1, 0] = d[1, 0]*2; d[0, -1] = d[0, -1]*2
    
    return(d)
    
        
def AV_diags(m):
    
    d = np.vstack([e(m-1), e(m-1)])
    d[1, 0] = d[1, 0]*2; d[0, -1] = d[0, -1]*2
    
    return d


def getGRAD(x, y):

    nx = np.size(x); nxc = nx-1; 
    ny = np.size(y); nyc = ny-1
    
    xc  = n2c(x)
    dxc = dnc(xc)

    yc  = n2c(y)
    dyc = dnc(yc)

    Dc2nx = ssp.spdiags(neumann_diags(dxc,nx),[-1,0],nx,nxc)
    Dc2ny = ssp.spdiags(neumann_diags(dyc,ny),[-1,0],ny,nyc)

    Gcx = ssp.kron(ssp.eye(nxc,nxc), Dc2ny)
    Gcy = ssp.kron(Dc2nx, ssp.eye(nyc,nyc))

    GRAD = ssp.vstack([Gcx, Gcy])
    
    return GRAD

    
def getDIV(x, y):
    
    nx = np.size(x); nxc = nx-1
    ny = np.size(y); nyc = ny-1
    
    dx  = np.diff(x);
    dy  = np.diff(y);
    
    e1 = -e(nxc)*1./dx
    e2 =  e(nxc)*1./dx
    diagsx = np.vstack([e1, e2])
    diagsy = np.vstack([-e(nyc)*1./dy, e(nyc)*1./dy])
    
    Dn2cx = ssp.diags(diagsx,[0,1], shape=(nxc,nx))
    Dn2cy = ssp.diags(diagsy,[0,1], shape=(nyc,ny))
   
    Dnx = ssp.kron(ssp.eye(nxc,nxc), Dn2cy)
    Dny = ssp.kron(Dn2cx, ssp.eye(nyc,nyc))

    DIV  = ssp.hstack([Dnx, Dny]); DIV = DIV.tocsr()
    
    return DIV
    
    
def getAV(nx, ny):

    nxc = nx-1; nyc = ny-1
    
    diagsx = AV_diags(nx)
    diagsy = AV_diags(ny)


    Ac2fx = .5 * ssp.spdiags(diagsx,[-1, 0],nx,nxc)
    Ac2fy = .5 * ssp.spdiags(diagsy,[-1, 0],ny,nyc)

    #AVy = ssp.kron(ssp.eye(nx,nx), Ac2fy); AVy = AVy.todense()
    #AVx = ssp.kron(Ac2fx, ssp.eye(ny-1,ny-1)); AVx = AVx.todense()
    #AV  = np.dot(AVy, AVx)

    AVy = ssp.kron(ssp.eye(nxc,nxc), Ac2fy); #AVy = AVy.todense()
    AVx = ssp.kron(Ac2fx, ssp.eye(nyc,nyc)); #AVx = AVx.todense()
    #AV  = ssp.vstack([AVx,AVy])

    #AVsig = 1. / (np.dot(AV,(1./Sigma)))
    #AVsig = np.reshape(AVsig, np.shape(X), order='F')
    
    return AVx, AVy
    
def get_hmat(dx, dy):
 
    nxc = nx-1; nyc = ny-1 

    DX, DY = np.meshgrid(dx, dy)
    h = DX * DY; h = np.ravel(h, order='F')
    hmat = np.tile(h, (nxc*nyc, 1))
    
    return hmat