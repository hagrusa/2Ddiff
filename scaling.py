import numpy as np
import matplotlib.pyplot as plt
from main import solver
import timeit


def make_D(nx, ny,keyword = 'ones', scale = 1.0):
    if keyword == 'ones':
        return np.ones((nx,ny))
    elif keyword == 'zeros':
        return np.zeros((nx,ny))
    elif keyword == 'random':
        return scale*np.random.rand(nx,ny)
    elif keyword == 'reactor':
        D_h20 = 0.142





#calculate scaling relations between cell size and number of iterations
#set omega = 1.1 and tolerance to 1e-5

a = 1.
b = 1.

sizes = [1.0, .5, .4, .25, .2, .1] #cell sizes
tolerance = 1e-5
omega = 1.2
iteration_list = []
time_list = []
sizes = [1.0, .5]
for i in sizes:
    dx = i
    dy = i
    nx = 2*a/dx   #number of cells in x and y directions
    ny = 2*b/dy
    n = int(nx)+1#size of mesh
    m = int(ny)+1

    D = np.ones((nx,ny))
    Sigma_a = np.ones((nx,ny))
    S = np.ones((n,m))

    start = timeit.default_timer()
    phi, iterations, error = solver( a, b, dx, dy, D, Sigma_a, S, tolerance, omega)
    stop = timeit.default_timer()

    iteration_list += [iterations]
    time_list += [stop - start]
    print i

print iteration_list
print time_list


#plt.plot(2*a/np.array(sizes), iteration_list)
#plt.show()



#find optimal omega

a = 1.
b=1.
dx = .2
dy = .2
nx = int(2*a/dx)   #number of cells in x and y directions
ny = int(2*b/dy)
n = int(nx)+1#size of mesh
m = int(ny)+1
D = np.ones((nx,ny))
Sigma_a = np.ones((nx,ny))
S = np.ones((n,m))

tolerance = 1e-6
omega_list = np.linspace(1.0,2.0, 5)

plt.figure()
for trial in range(0,5):
    print trial
    D = np.random.rand(nx,ny)
    Sigma_a = np.random.rand(nx,ny)
    S = np.random.rand(n,m)
    iteration_list = []
    for i in omega_list:
        phi, iterations, error = solver(a, b, dx, dy, D, Sigma_a, S, tolerance, i)
        iteration_list += [iterations]
    plt.plot(omega_list, iteration_list)
plt.show()













