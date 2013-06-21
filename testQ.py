# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 06:13:51 2013

@author: ben
"""
import ModOps as mo


def get_Q(xint, x):
    
    def Q1d(xint, x):
       dx = np.diff(x)
       Qn = np.zeros((np.size(xint), np.size(x)))
    
       for i in range(0, np.size(xint)):
          xdif = x - xint[i]
          ind = np.where(xdif == min(xdif[xdif >= 0]))
          Qn[i][ind[0][0]] = xdif[ind] / dx[ind[0][0] - 1]
          Qn[i][ind[0][0] - 1] = 1 - (xdif[ind] / dx[ind[0][0] - 1])
       return Qn
       
    Qx = Q1d(xint[0][:], x[0][:])
    Qy = Q1d(xint[1][:], x[1][:])
    
    QxOp, QyOp = kronOp(np.size(x[0][:]), np.size(x[1][:]), Qx, Qy, 'comb')
    
    return QxOp, QyOp
    
Qx, Qy = get_Q(np.array([xdat, ydat]), np.array([xc, yc]))