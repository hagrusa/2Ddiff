import numpy as np
import matplotlib.pyplot as plt
import construct as con
from SOR import SOR

def solver(a,b,dx,dy,D,Sigma_a, v_Sigma_f, S, tolerance, omega):
    #this current code only works with reflecting boundary conditions on all sides,
    #I am having trouble with the vacuum boundary conditions

    #a is x dimension in cm [-a,a]
    #b is y dimension in cm [-b,b]
    #dx is size of cells in x direction
    #dy is size of cells in y direction
    #current code only works when a = b and dx=dy
    #D is diffusion constant [  ], must be a 2D array of dimension (2*a/dx, 2*b/dy)
    #Sigma_a is abosorption cross section [  ],  must be a 2D array of dimension (2*a/dx, 2*b/dy)
    #S is source, must be 2 array of dimesion (2*a/dx+1, 2*b/dy+1)
    nx = 2*a/dx   #number of cells in x and y directions
    ny = 2*b/dy
    n = int(nx)+1#size of mesh
    m = int(ny)+1

    A = con.construct_A(n,m, D, Sigma_a, v_Sigma_f, dx, dy)
    b = con.construct_b(S,n,m)

    phi, iterations, error = SOR(A, b, np.ones((n*m)), tolerance, omega)

    return phi, iterations, error












