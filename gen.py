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
As = 1

def initial_Power_Spectrum_BKKS(K):
    '''spatial part of Power spectrum (units = Mpc^3) at wavenumber k (units = Mpc^-1)'''
    return ( As * np.power(K,ns) *  transfer_Function_BKKS(K) ** 2)


####"
# n = np.zeros((200,200), dtype=complex)
# n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))
# im = np.fft.ifftn(n).real
# plt.imshow(im)
# plt.show()

# n = np.zeros((200,200), dtype=complex)
# n = np.exp(1j*np.random.uniform(0, 2*np.pi, (200, 200))) #génération d'un bruit blanc
# im = np.fft.ifftn(n).real
# plt.imshow(im)
# plt.show()

##### 2D fft
N = 200
r = np.random.normal(0, 1, size= (N,N)) #200x200 Mpc box
delta_r = 100 #Mpc ie entre 2 points on met 100Mpc
delta_k = 1/delta_r # à corriger avec un 2pi ?
R = np.fft.fftn(r)
K = np.sqrt(np.array([[(k1*delta_k)**2+(k2*delta_k)**2 for k2 in range(N)] for k1 in range(N)]))
sqP = np.sqrt(initial_Power_Spectrum_BKKS(K))
sqPR = np.multiply(sqP,K)
sqpr = np.fft.ifftn(r).real

# Matricial plot
# plt.imshow(sqpr)
# plt.imshow(r)
# plt.imshow(R)

# 2D plot
# x = np.linspace(1,N,N)
# y = x
# x,y = np.meshgrid(x,y)
# # plt.contourf(x,y,sqpr,2)
# plt.contourf(x,y,sqpr,[0.001*i for i in range(5)])
# plt.show()

##### 3D fft
# N = 200
# r = np.random.normal(0, 1, size= (N,N,N)) #200x200 Mpc box
# delta_r = 100 #Mpc ie entre 2 points on met 100Mpc
# delta_k = 1/delta_r # à corriger avec un 2pi ?
# R = np.fft.fftn(r)
# K = np.sqrt(np.array([[(k1*delta_k)**2+(k2*delta_k)**2 for k2 in range(N)] for k1 in range(N)]))
# sqP = np.sqrt(initial_Power_Spectrum_BKKS(K))
# sqPR = np.multiply(sqP,K)
# sqpr = np.fft.ifftn(r).real
# print(np.max(sqpr))
#

# Il faudra trouver le bon module pour afficher une isosurface en 3D


#     Z = np.array([[temp.projected_HMF(np.power(10,M),z)*(np.pi/180)**2 for M in X] for z in Y])
#     X, Y = np.meshgrid(X, Y)
#     fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#     surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                            linewidth=0, cmap = cm.gnuplot)