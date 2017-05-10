#not all of these packages are used in this script, they are just here in case
import numpy as np
import matplotlib.pyplot as plt
import timeit
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import pylab
from numpy import ma
from matplotlib import colors, ticker, cm
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


#these are scripts I wrote
import construct as con
from main import solver
import make_plots


a = 1. #x axis goes from [-a,a] in centimeters, for now a must be equal to b
b=1.   #y axis goes from [-b,b] in centimeters

dx = .02 #cell length in x direction, must be same as dy for now. [cm]
dy = .02 #cell length in y direction

nx = int(2*a/dx)   #number of cells in x and y directions
ny = int(2*b/dy)
n=nx+1   #number of nodes in x and y directions
m=ny+1

D = np.empty((nx,ny)) #diffusion constant array, must be an array of shape (nx,ny) in centimeters
D[:,:] = 0.142 # D for water, [cm]   #we are making the entire space full of water

Sigma_a = np.empty((nx,ny)) #absorption cross section, must be an array of shape (nx,ny), has units of 1/cm
Sigma_a[:,:] = 0.022 # absorption cross-x for water, [1/cm]

v_Sigma_f = np.zeros((nx,ny)) #fission term, must be array of size (nx,ny)
v_thermal = 2.2e5 # velo2city of thermal neutrons, cm/s
Sigma_f = 0.0 #macroscopic fission xsection cm^{-1}. we are setting this to zero, it is highly unlikely that water will fission
v_Sigma_f[:,:] = v_thermal * Sigma_f #filling array with zeros, (no fission). previous two lines were just for example, but dont do anything in this example


S = np.zeros((n,m)) #source term, must be of shape (n,m). has units of [neutrons/cm/s] (we are in 2D, otherwise it would be [neutrons/cm^2/s])
S[n//2,m//2] = 1e9 #10^9 neutrons /cm/s, #putting point source in middle of water



#main.solver() builds the matrix and solves it using SOR.py
#it takes the following arguements
#(x length [cm], y length [cm], cell x length [cm], cell y length [cm], diffusion constant array,
#   absorption x-section array, fission x-section array, source array, tolerance: relative, SOR parameter: omega)
#phi_vector, iterations, error = solver(a,b,dx,dy, D, Sigma_a, v_Sigma_f, S, 1e-5, 1.7)

#reshapes our solution back into a 2D array for plotting
#phi = con.x_to_phi(phi_vector, n, m)

#find optimal omega
#make_plots.optimize_omega_plot(a,b, dx, dy, D, Sigma_a, v_Sigma_f, S, omegas = 5, tolerance = 1e-4)

#make contour plot
#make_plots.plot_phi_2D(phi, a, b, dx, dy, title = r'$\phi(x,y)$ for Point Source', filename = 'point_source_2D_big2.pdf')

#make surface plot
#make_plots.plot_phi_3D(phi, a ,b, dx, dy)

make_plots.scaling(a,b)



