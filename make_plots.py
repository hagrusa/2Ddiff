import numpy as np
import matplotlib.pyplot as plt
from main import solver
import timeit
import construct as con

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import pylab
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib import colors, ticker, cm

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



#plot phi array, contour plot
def plot_phi_2D(phi,a,b,dx,dy, title = "Make a Title!!", filename = "test.pdf"):

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(-b, b + dy, dy), slice(-a, a + dx, dx)]

    plt.pcolor(x,y,phi,norm=colors.LogNorm(vmin=phi.min(), vmax=phi.max()))
    cb = plt.colorbar()
    cb.set_label(r"$\bf{\phi(x,y) \quad [cm^{-1}s^{-1}]}$", fontsize = 16, rotation = 270)
    plt.title(title, fontsize = 18)
    plt.xlabel('x [cm]', fontsize = 16)
    plt.ylabel('y [cm]', fontsize = 16)
    if filename == 'test.pdf':
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

#plot 3d surface, this opens the image in your web browser where you can zoom in and look around
def plot_phi_3D(phi, a, b, dx, dy, title = "make title", filename = 'test.pdf'):
    #needs work
    y, x = np.mgrid[slice(-b, b + dy, dy),
                slice(-a, a + dx, dx)]

    data = [go.Surface(x = x, y = y, z = phi)]
    fig = go.Figure(data=data)
    plot(fig)


#find optimal omega
def optimize_omega_plot(a, b, dx, dy, D, Sigma_a, v_Sigma_f, S, omegas = 10, tolerance = 1e-4, filename = 'test.pdf'):
    omega_list = np.linspace(1.0,1.9, omegas)
    iteration_list = []
    time_list = []
    for i in omega_list:
        start = timeit.default_timer()
        phi, iterations, error = solver(a, b, dx, dy, D, Sigma_a, v_Sigma_f, S, tolerance, i)
        stop = timeit.default_timer()
        time_list += [stop-start]
        iteration_list += [iterations]
        print "omega = " + str(i) +" | iterations: " + str(iterations) + " | time: " + str(stop-start)


    fig, ax1 = plt.subplots()

    handle1, = ax1.plot(omega_list, iteration_list, 'b', label = "Iterations")
    ax1.set_xlabel(r'$\omega$', fontsize = 18)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Number of iterations', fontsize = 18, color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    handle2, = ax2.plot(omega_list, time_list, 'g', label = "Time")
    ax2.set_ylabel('Time to solve A [s]', fontsize=18, color='g')
    ax2.tick_params('y', colors='g')
    plt.title('Optimizing Omega', fontsize=20)
    plt.xlim(0.9, 2.0)
    plt.legend(handles=[handle1, handle2])

    if filename == 'test.pdf':
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()



#calculate scaling relations between cell size and number of iterations
def scaling(a, b):
    sizes = np.array([1.0, .5, .4, .25, .2, .1, .05]) #cell sizes
    tolerance = 1e-4
    omega = 1.66 #optimal omega for water problem
    iteration_list = []
    time_list = []


    #sizes = [1.0, .5]
    for i in sizes:
        dx = i
        dy = i
        nx = 2*a/dx   #number of cells in x and y directions
        ny = 2*b/dy
        n = int(nx)+1#size of mesh
        m = int(ny)+1

        D = np.empty((nx,ny))
        D[:,:] = 0.142 # D for water, [cm]

        Sigma_a = np.empty((nx,ny))
        Sigma_a[:,:] = 0.022 # absorption cross-x for water, [1/cm]

        S = np.zeros((n,m))
        S[n//2,m//2] = 1e9

        v_Sigma_f = np.zeros((nx,ny))


        start = timeit.default_timer()
        phi, iterations, error = solver( a, b, dx, dy, D, Sigma_a, v_Sigma_f, S, tolerance, omega)
        stop = timeit.default_timer()

        iteration_list += [iterations]
        time_list += [stop - start]
        print i



    plt.xlabel('Cell Size [cm]', fontsize = 18)
    plt.ylabel('Number of Iterations', fontsize = 18)
    plt.plot(sizes,iteration_list,'b') # against 1st x, 1st y
    plt.xlim(0.0, 1.0)
    plt.twinx()
    plt.ylabel('Time so solve A', fontsize = 18)
    plt.plot(sizes,time_list,'g') # against 1st x, 2nd y
    plt.xlim(0.0,1.0)
    plt.twiny()
    plt.xlabel('Number of cells', fontsize = 18)
    plt.plot(2*a/sizes, time_list,'g', alpha = 0.0) # against 2nd x, 2nd y
    plt.gca().invert_xaxis()
    plt.xlim(2*a/np.min(sizes), 2*a/np.max(sizes))
    plt.savefig('scaling2.pdf')
    plt.show()



"""


fig, ax1 = plt.subplots()




ax1.plot(sizes, iteration_list, 'b.')
ax1.set_xlabel('Cell size [cm]', fontsize = 18)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Number of iterations', fontsize = 18, color='b-')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(sizes, time_list, 'k.')
ax2.set_ylabel('Time to solve A', fontsize=18, color='k-')
ax2.tick_params('y', colors='k')
plt.title('Cell Size scalings', fontsize=20)
plt.xlim(0, 1.1)
plt.savefig('size_scaling_water.pdf')
plt.close()











"""






