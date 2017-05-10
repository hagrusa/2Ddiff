import copy
import numpy as np


def SOR(A, b, x_guess, tolerance, omega):
    error = 1.0
    iteration = 0

    n = len(x_guess)
    new_x = np.zeros(n)
    old_x = x_guess
    while error > tolerance: #iterate until error is within tolerance
        iteration+=1
        for i in range(0, n):
            summation = 0.0
            for j in range(0,n):
                if j < i:
                    summation += A[i,j]*new_x[j]
                elif j > i:
                    summation += A[i,j]*old_x[j]
            new_x[i] = (1-omega)*old_x[i] + (omega/A[i,i])*(b[i] - summation)
        error = np.linalg.norm(np.abs(new_x - old_x),2) / np.linalg.norm(new_x,2) # relative error
        old_x = copy.deepcopy(new_x)
    return new_x, iteration, error #returns solution, number of iterations, relative error






