import numpy as np
import math
from numpy import *
import sys
import sympy as sy

def armijo(gk,dk,x0,p,sigma):
    return fun(x0+p*dk) < fun(x0) + sigma * p * np.dot(gk,dk)

def ModifiedNewtonMethodArmijo(f,g,H,x0):
    x0_vals=[]
    x1_vals=[]
    f_vals = []
    x0_vals.append(x0[0])
    x1_vals.append(x0[1])
    f_vals.append(f(x0))
    
    maxNumofIter = 500
    n= np.shape(x0)[0]
    rho = 0.55
    sigma = 0.4
    tau  = 0.0 
    k=0
    epsilon =1e-5
    while k < maxNumofIter:
        gk = g(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        muk = np.power(np.linalg.norm(gk),1+tau)
        Hk = H(x0)
        Ak = Hk + muk*np.eye(n)
        dk = -1.0 *np.linalg.solve(Ak,gk)
        m=0
        mk=0
        while m < 20:
            if armijo(gk,dk,x0,rho**m,sigma):
                mk = m
                break
            m+=1
        x0 += rho**mk  * dk
        x0_vals.append(x0[0])
        x1_vals.append(x0[1])
        f_vals.append(f(x0))
        k+=1
    print("==========  Armijo")
    print("xk= ",x0)
    print("f(xk)= ",round(f(x0),3))
    print("No of iterations =", k)
    return x0_vals,x1_vals, f_vals
    
f_counter = 0
g_counter = 0
H_counter = 0
x=sy.IndexedBase('x')

n=2 

fexpr = (2*x[0]+1) **2 + x[1] **2
gexpr = [sy.diff(fexpr,x[i]) for i in range(n)]
Hexpr = [[sy.diff(g,x[i]) for  i in range(n)] for g in gexpr]

flamdify = sy.lambdify(x,fexpr,"numpy")
glambdify = [sy.lambdify(x,gf,"numpy") for gf in gexpr]
Hlambdify = [[sy.lambdify(x,hf,"numpy") for hf in Hs] for Hs in Hexpr]

def fun(x):
    global f_counter
    f_counter+=1
    return flamdify(x)

def gfun(y):
    global g_counter
    g_counter+=1
    return np.array([gf(y) for gf in glambdify])

def hess(y):
    global H_counter
    H_counter+=1
    return np.array([[gf(y) for gf in Hs] for Hs in Hlambdify])

x0_vals,x1_vals,f_vals=ModifiedNewtonMethodArmijo(fun,gfun,hess,np.zeros(n)+np.array([5,6]))
                  