import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy.integrate as intg

###
Ho = 70  # Hubble constant today (units = km.s^-1.Mpc^-1)
Om = 0.3  # Normalized density parameter for matter (units = 1)
Or = 0  # Normalized density parameter for radiations (units = 1)
Ov = 0.7  # Normalized density parameter for vacuum (units = 1)
OT = 1  # Normalized total density parameter (units = 1)
sigma8 = 0.8  # RMS density parameter at 8 h^-1 Mpc (units = 1)
ns = 0.96  # Inflation exponent in Power spectrum law (units = 1)
k = 0  # Curvature -1, 0 ( <=> Omega_T = 1) or 1 (units = 1)
h = 0.7


def transfer_Function_BKKS(k,Om = 0.3,h= 0.7):
    '''Transfer function at wavenumber k (units Mpc^-1)'''
    theta = 1
    q = k * theta ** 0.5 / (Om * h ** 2)  # Mpc
    return (np.log(1 + 2.34 * q) / (2.34 * q)) * np.power(1 + 3.89 * q + np.power(16.1 * q, 2) + np.power(5.46 * q, 3) + np.power(6.71 * q, 4), -0.25)


def window(y):
    '''Window function in Fourier space, the product with which allows to get rid of low values of radius or mass'''
    return (3 * (np.sin(y) / y - np.cos(y)) / np.power(y, 2))

# As = sigma8 ** 2 * (2 * m.pi) ** 3 / (4*m.pi*intg.quad(lambda K : K **  ns *  transfer_Function_BKKS(K) ** 2 * abs( window(K * 8/ h))** 2 * K ** 2, 0, m.inf,limit = 100)[0])
As = 6027309.4271296235

def initial_Power_Spectrum_BKKS(K):
    '''spatial part of Power spectrum (units = Mpc^3) at wavenumber k (units = Mpc^-1)'''
    return (As * np.multiply(np.power(K,ns),np.power(transfer_Function_BKKS(K),2)))




#GENERATE WHITE NOISE
N = 40
delta_r = 100 #Mpc ie entre 2 points on met 1000Mpc
delta_k = 1/delta_r # à corriger avec un 2pi ?
r = np.random.normal(0, 1, size= (N,N,N))

# FOURIER SPACE
R = np.fft.fftn(r,axes = [0,1,2])
# R = np.fft.rfftn(r,axes = [0,1,2])
# print(R.shape)
K = np.sqrt(np.array([[[(k1*delta_k)**2+(k2*delta_k)**2+(k3*delta_k)**2 for k3 in range(N)] for k2 in range(N)] for k1 in range(N)]))
P = initial_Power_Spectrum_BKKS(K)
sqP = np.nan_to_num(np.sqrt(P))
sqPR = np.multiply(sqP,R)

# BACK TO REAL SPACE
p = np.fft.ifftn(P,axes = [0,1,2])
# p = np.fft.irfftn(P)
sqpr = np.fft.ifftn(sqPR,axes = [0,1,2])
# sqpr = np.fft.irfftn(sqPR)

#TOMOGRAPHIE :
p2D = p[int(N/2),:,:]
sqpr2D = sqpr[int(N/2),:,:]

# 2D plot (tomographie milieu)
x = np.linspace(1,N,N)
y = x
x,y = np.meshgrid(x,y)
plt.contourf(x,y,sqpr2D)
plt.colorbar()
plt.show()



