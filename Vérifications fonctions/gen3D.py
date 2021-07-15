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
noise = np.random.normal(0, 1, size= (N,N,N))
print(noise)
# plt.imshow(noise[N//2,:,:])
# plt.show()
# FOURIER SPACE
# NOISE = np.fft.fftn(noise,axes = [0,1,2])
NOISE = np.fft.rfftn(noise,axes = [0,1,2])
# print(NOISE.shape)
noise = np.fft.irfftn(NOISE[N//2,:,:])
print(noise)
plt.imshow(noise)
plt.show()
Kx = np.fft.fftfreq(N,d=delta_r)
Rx = np.fft.fftfreq(N,d=delta_k)
print(Kx)
print(Rx)
# Kx
# K = np.sqrt(np.array([[[(k1*delta_k)**2+(k2*delta_k)**2+(k3*delta_k)**2 for k3 in range(N)] for k2 in range(N)] for k1 in range(N)]))
K = np.sqrt(np.array([[[(k1*delta_k)**2+(k2*delta_k)**2+(k3*delta_k)**2 for k3 in Kx] for k2 in Kx] for k1 in Kx]))
P = initial_Power_Spectrum_BKKS(K) ### OK JE CROIS QUE L'ORDRE DES K EST A REVOIR !!
sqP = np.nan_to_num(np.sqrt(P))
# sqPNO = np.multiply(sqP,NOISE)

# BACK TO SPATIAL SPACE
# p = np.fft.ifftn(P)
# print(p)
# p = np.fft.irfftn(P)
# sqpno = np.fft.ifftn(sqPNO,axes = [0,1,2])
# sqpno = np.fft.irfftn(sqPNO)

#TOMOGRAPHIE :
# p2D = p[int(N/2),:,:]
# sqpno2D = sqpno[int(N/2),:,:]
# # print(p2D)
# # 2D plot (tomographie milieu)
# x = np.linspace(1,N,N)
# y = x
# x,y = np.meshgrid(x,y)
# plt.contourf(x,y,sqpno2D)
# plt.colorbar()
# plt.show()




