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

def transfer_Function_BKKS( K, theta=1):
    '''Transfer function at wavenumber k (units Mpc^-1)'''
    q = k * theta ** 0.5 / ( Om *  h ** 2)  # Mpc
    if k == 0:
        return (1)
    else:
        return ((m.log(1 + 2.34 * q) / (2.34 * q)) * (
                    1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** (-0.25))


def window( y):
    '''Window function in Fourier space, the product with which allows to get rid of low values of radius or mass'''
    try:
        if y < 1e-7:
            return (1)
        elif y == m.inf:
            return (0)
        else:
            return (3 * (m.sin(y) / y - m.cos(y)) / y ** 2)
    except ValueError:
        print(y)

# As = sigma8 ** 2 * (2 * m.pi) ** 3 / (4*m.pi*intg.quad(lambda K : K **  ns *  transfer_Function_BKKS(K) ** 2 * abs( window(K * 8/ h))** 2 * K ** 2, 0, m.inf,limit = 100)[0])
As = 1
def initial_Power_Spectrum_BKKS(K):
    '''spatial part of Power spectrum (units = Mpc^3) at wavenumber k (units = Mpc^-1)'''
    return ( As * K **  ns *  transfer_Function_BKKS(K) ** 2)


####"
# n = np.zeros((200,200), dtype=complex)
# n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))
# im = np.fft.ifftn(n).real
# plt.imshow(im)
# plt.show()

n = np.zeros((200,200), dtype=complex)
n = np.exp(1j*np.random.uniform(0, 2*np.pi, (200, 200))) #génération d'un bruit blanc
im = np.fft.ifftn(n).real
plt.imshow(im)
plt.show()

# n = [[P(i)] for i in range 200]
# n = np.exp(1j*np.random.uniform(0, 2*np.pi, (200, 200))) #génération d'un bruit blanc
# im = np.fft.ifftn(n).real
# plt.imshow(im)
# plt.show()