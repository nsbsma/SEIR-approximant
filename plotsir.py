#This uses SIR_Approximant.py to plot the asymptotic approximant to the SIR equations,
#given in Barlow & Weinstein, Physica D 408, 1 (2020) (arxiv.org/abs/2004.07833)
# Code written by J. Van Dyke and N. Barlow 6/17/2020

import numpy as np
import matplotlib.pyplot as plt
from SIR_Approximant import ApproximantCoefficientsSIR
from math import e

# set parameters, example given below to reproduce figure 2b
N=25 #number of terms in approximant
S0=254 #initial susceptible population
I0 = 7 #initial infected population
r = 0.0178 #SIR parameter
alpha = 2.73 #SIR parameter
duration = 6 # duration
dt = 0.01  # time resolution


# return A, kappa, Sinf
A, kappa, Sinf, a0, a = ApproximantCoefficientsSIR(N,alpha,r,S0,I0)

# calculate Susceptible eq (12a)
def calcSA(t, N, A, kappa, Sinf):
    sum = 1
    for i in range(0, N): # starts on A[0]
        sum = sum + A[i]*np.power(e, (i+1)*kappa*t)
    return Sinf/sum

# step through time
nsteps=int(duration/dt) # Number of steps
S = np.zeros([nsteps])
I  = np.zeros([nsteps])
R  = np.zeros([nsteps])
for i in range(0,nsteps):
    S[i] = calcSA(i*dt, N, A, kappa, Sinf)
    I[i] = alpha*np.log(S[i]/S0)/r - S[i] + S0 + I0 #equation (4)
    R[i] = I0 + S0 - I[i] -S[i] # using S+I+R=constant

# Plot the data on three separate curves for S(t), I(t) and R(t)
t = np.linspace(0, duration, nsteps)
plt.plot(t,S,label='S')
plt.plot(t,I,label='I')
plt.plot(t,R,label='R')
plt.xlabel('t')
plt.legend()
plt.show()

