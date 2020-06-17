#This provides the function [A,Sinf]=ApproximantCoefficientsSIR(N,alpha,r,S0,I0)
#which computes the A_n coefficients and Sinf needed for the N-term SIR approximant
#given as equation (12a) in Barlow & Weinstein, Physica D 408, 1 (2020) (arxiv.org/abs/2004.07833)
#The inputs correspond to the SIR parameters and initial conditions as
#specified by equation (1) in the preprint.
# Code written by J. Van Dyke and N. Barlow 6/17/2020

import numpy as np
from scipy.optimize import curve_fit, fsolve
import scipy.special

#Vandermonde matrix inverter
#Input is np.array
# Call this once to fill matrix
def VDMinv(N):
    #zero NxN matrix
    L = np.zeros((N,N))
    U = np.zeros((N,N))
    L[0][0] = 1
    im1=0
    jm1=0
    for i in range(1,N+1):
        for j in range(1,N+1):
            im1 = i-1
            jm1 = j-1
            if j*i !=1 and i>=j:
                prod=1
                for k in range(1,i+1):
                    if j !=k:
                        prod=prod/(j-k)
                L[im1][jm1] = prod
            if j==i:
                U[im1][jm1] = 1
            elif j !=1:
                if i==1:
                    U[im1][jm1] = -U[im1][j-2]*(jm1)
                else:
                    U[im1][jm1] = U[i-2][j-2] -U[im1][j-2]*(jm1)

    #multiply U*L then transpose array
    f = (U@L).transpose()
    return f



#function [A,Sinf]=ApproximantCoefficientsSIR(N,alpha,r,S0,I0)
#This computes the A_n coefficients and Sinf needed for the N-term SIR approximant
#given as equation (12a) in arxiv.org/abs/2004.07833
#The inputs correspond to the SIR parameters and initial conditions as
#specified by equation (1) in the preprint. 
def ApproximantCoefficientsSIR(N,alpha,r,S0,I0):
    beta=alpha*np.log(S0)-r*(S0+I0)
    a0=S0
    b0=np.log(a0)
    at0=1.0/a0
    a = np.zeros(N)
    at = np.zeros(N)
    b = np.zeros(N)    
    a[0]=beta*a0+a0*(r*a0-alpha*b0)
    for n in range(1,N):
        nm1=n-1
        sum1=a[nm1]*at0
        for j in range(0,nm1):
            sum1=sum1+a[j]*at[nm1-j-1]
        at[nm1]=-sum1/a0
        sum2=n*a[nm1]*at0
        for j in range(0,nm1):
            sum2=sum2+(j+1)*a[j]*at[nm1-j-1]
        b[nm1]=sum2/n
        asum=a0*(r*a[nm1]-alpha*b[nm1])+a[nm1]*(r*a0-alpha*b0)
        for j in range(0,nm1):
            asum=asum+a[j]*(r*a[nm1-j-1]-alpha*b[nm1-j-1])
        a[n]=(asum+beta*a[nm1])/(n+1)

    Sinf = fsolve(lambda Sinf: alpha/r*np.log(Sinf/S0)+(S0-Sinf)+I0, 0.1)
    kappa=r*Sinf-alpha
    C=1.0/kappa
    aa0=at0*Sinf
    aa=at*Sinf

    M = np.zeros((N,N))
    M[1,:] = np.ones(N)
    b[0] = aa0-1
    for j in range(1,N):
        M[j,:]= np.power(np.arange(1,N+1),j)
        b[j]=aa[j-1]*scipy.special.factorial(j)*np.power(C,j)
    Minv=VDMinv(N)
    A=Minv@b.transpose()
    return A, kappa, Sinf, a0, a

