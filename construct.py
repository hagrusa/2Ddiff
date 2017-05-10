import numpy as np


def construct_A(n,m, D, Sigma_a, v_Sigma_f, dx, dy):

    A = np.array(np.zeros((n*m,n*m)))
    for i in range(0,n):
        for j in range(0,m):
            a_matrix = np.zeros((n,m))
            for ii in range(0,n):
                for jj in range(0,m):
                    a_matrix[ii,jj] = calculate_a(ii , jj , i , j, n, m, D, Sigma_a, v_Sigma_f, dx, dy)
            A[m*i:m*i +m, n*j:n*j + n] = a_matrix
    return A


def calculate_a(ii,jj,i,j,n,m,D,Sigma_a, v_Sigma_f, dx,dy):
    if jj==0: #left
        if j==0: #bottom left(vacuum)
            a_left = 0.0
            a_right = -(D[jj,j]*dy)/(2*dx)
            a_bottom = 0.0
            a_top = -D[jj,j]*dx/(2*dy)
            a_center = 0.0
            sigma_a = 0.25*dx*dy * (Sigma_a[jj,j])
            v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj,j])
        elif j==m-1: #top left(vacuum)
            a_left = 0.0
            a_right = -D[jj,j-1]*dy/(2*dx)
            a_bottom =-(D[jj,j-1]*dx)/(2*dy)
            a_top = 0.0
            a_center = 0.0 ####
            sigma_a = 0.25*dx*dy * Sigma_a[jj,j-1]
            v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj,j-1])

        else:
            a_left = 0.0
            a_right = -(D[jj,j-1]*dy + D[jj,j]*dy)/(2*dx)
            a_bottom = -D[jj,j-1]*dx/(2*dy)
            a_top = -D[jj,j]*dx/(2*dy)
            a_center = 0.0 #vacuum boundary ??? ****
            sigma_a = 0.25*dx*dy * (Sigma_a[jj,j-1] + Sigma_a[jj,j])
            v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj,j-1] + v_Sigma_f[jj,j])

    elif jj==n-1:#right
        if j==0: #bottom right (vacuum)
            a_left = -(D[jj-1,j]*dy)/(2*dx)
            a_right = 0.0
            a_bottom = 0.0
            a_top = -D[jj-1,j]*dx/(2*dy)
            a_center = 0.0
            #a_center = Sigma_a[jj-1,j-1] - (a_left+a_right+a_bottom+a_top)
            sigma_a = 0.25*dx*dy * Sigma_a[jj-1,j]
            v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj-1,j])

        elif j==m-1: #top right (reflecting)
            a_left = -D[jj-1,j-1]*dy/(2*dx)
            a_right = 0.0
            a_bottom = -D[jj-1,j-1]*dx/(2*dy)
            a_top = 0.0
            a_center = Sigma_a[jj-1,j-1] - (a_left+a_right+a_bottom+a_top)
            sigma_a = 0.25*dx*dy*Sigma_a[jj-1,j-1]
            v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj-1,j-1])

        else:
            a_left = -(D[jj-1,j-1]*dy + D[jj-1,j]*dy)/(2*dx)
            a_right = 0.0
            a_bottom = -D[jj-1,j-1]*dx/(2*dy)
            a_top = -D[jj-1,j]*dx/(2*dy)
            a_center = Sigma_a[jj-1,j-1] - (a_left+a_right+a_bottom+a_top)
            sigma_a = 0.25*dx*dy *(Sigma_a[jj-1,-1] + Sigma_a[jj-1,j])
            v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj-1,j-1] + v_Sigma_f[jj-1,j])


    elif j==0: #bottom
        a_left = -D[jj-1,j]*dy/(2*dx)
        a_right = -D[jj,j]*dy/(2*dx)
        a_bottom = 0.0
        a_top = -(D[jj-1,j]*dx + D[jj,j]*dx)/(2*dy)
        a_center = 0.0
        sigma_a = 0.25*dx*dy*(Sigma_a[jj-1,j] +Sigma_a[jj,j])
        v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj-1,j] + v_Sigma_f[jj,j])


    elif j==m-1: #top
        a_left = -D[jj-1,j-1]*dy/(2*dx)
        a_right = -D[jj,j-1]*dy/(2*dx)
        a_bottom = -(D[jj-1,j-1]*dx +D[jj,j-1]*dx)/(2*dy)
        a_top = 0.0
        a_center = Sigma_a[jj-1,j-1] - (a_left+a_right+a_bottom+a_top)
        sigma_a = 0.25*dx*dy*(Sigma_a[jj-1,j-1] +Sigma_a[jj,j-1])
        v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj-1,j-1] + v_Sigma_f[jj,j-1])

    else: #no boundary
        a_left   = - (dy*D[jj-1   , j-1  ] + dy*D[jj-1   , j]) / (2*dx)
        a_right  = - (dy*D[jj , j-1  ] + dy*D[jj , j]) / (2*dx)
        a_bottom = - (dx*D[jj-1   , j -1 ] + dx*D[jj , j -1 ]) / (2*dy)
        a_top    = - (dx*D[jj-1   , j] + dx*D[jj , j]) / (2*dy)
        a_center = Sigma_a[jj-1,j-1] - (a_left+a_right+a_bottom+a_top)
        sigma_a = 0.25*dx*dy *(Sigma_a[jj-1,j-1]+ Sigma_a[jj,j-1] +Sigma_a[jj-1,j] +Sigma_a[jj,j])
        v_sigma_f = 0.25*dx*dy * (v_Sigma_f[jj-1,j-1] + v_Sigma_f[jj,j-1] + v_Sigma_f[jj-1,j] + v_Sigma_f[jj,j])

    a_center = sigma_a + v_sigma_f - (a_left+a_right+a_bottom+a_top) #make everything reflecting boundary for now, vacuum boundary conditions arent working

    if i==j:
        if ii == jj:
            return a_center
        elif ii - jj ==  1:
            return a_left
        elif ii - jj == -1:
            return a_right
        else:
            return 0.0
    elif i - j == 1:
        if ii==jj:
            return a_bottom
        else:
            return 0.0
    elif i - j == -1:
        if ii==jj:
            return a_top
        else:
            return 0.0
    else:
        return 0.0

def construct_b(S,n,m):
    b = np.empty(0)
    for j in range(0,m):
        for i in range(0,n):
            b = np.append(b, np.array([S[i,j]]))
    return b

def x_to_phi(x,n,m):
    return np.reshape(x,(n,m))

