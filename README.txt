See sample_deck.py to see how code works.

input decks are python scripts.
In the script, you specify the parameters of the problem
The input script creates arrays containing the diffusion constant, absorption cross section, neutron energy/velocity, fission cross section, and source
main.py then takes this information creates the matrices A and b, then calls the SOR subroutine to solve the system of equations.

the solution is then returned along with the number of iterations required to solve, and the relative error of the solution
the x_to_phi function in construct.py then converts the 1D array back into a 2D array for plotting

make_plots also has some functions to create 2 and 3D plots of the solutions. note that the functions may require custom editing depending on what how you want to plot phi(x,y)



Current issues/limitations:
This code only works with all reflecting boundary conditions. For some reason, I cann't get the vacuum boundary conditions to work - it is probably an easy fix if I had another set of eyes looking at the code

Also, the x and y dimension and cell size must be the same. Given more time, I could easily have fixed this to make the code more robust.

To simulate real materials, you have to look up the diffusion constant, and absorption and fission x-sections  for the neutron energy in mind and calculate them by hand. If i had more time i would have built a function that can calculate these things for you using existing data bases.

