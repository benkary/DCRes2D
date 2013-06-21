# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:54:17 2013

@author: ben
"""
import ModOps as mo
import numpy as np

nx = 21.  ; nxc = nx - 1
ny = 11.  ; nyc = ny - 1

x   = np.linspace(0,1,nx)
y   = np.linspace(0,1,ny)

xc  = mo.n2c(x)
yc  = mo.n2c(y)

 

xdat = xc[13:20]
ydat = np.zeros((np.shape(xdat)))

Qx, Qy = mo.get_Q(np.array([xdat, ydat]), np.array([xc, yc]))