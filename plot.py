# from matplotlib import cbook
from matplotlib import cm
# from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import Cosmological_tools as cosmo
import numpy as np

def plotHMFatZ(z):
    # X = M in solar Masses
    X = np.linspace(14,16,201)
    temp = cosmo.Cosmology()
    Y = np.ravel([temp.projected_HMF(np.power(10,M),z)*(np.pi/180)**2 for M in X])
    plt.plot(X,Y)
    plt.title('HMF per deg^2 at redshift '+str(z))
    plt.xlabel('log M[solarMasses]')
    plt.ylabel('HMF per deg^2')
    plt.grid()
    plt.show()
# plotHMFatZ(1)

def plotHMFatM(M):
    if M == 1e14:
        X, Yref = cosmo.readtxt('hmfz_PS.txt')
        plt.plot(X,Yref, label = 'Ref', color = 'red')
    else:
        X = np.linspace(0,3,501)
    temp = cosmo.Cosmology()
    Y = np.ravel([temp.projected_HMF(M,z)*(np.pi/180)**2 for z in X])
    plt.plot(X,Y, label = 'Mine', color = 'blue', linestyle = '--')
    plt.title('HMF per deg^2 at '+format(M,'.2e')+' solar masses')
    plt.xlabel('Redshift')
    plt.ylabel('HMF per deg^2')
    plt.legend()
    plt.grid()
    plt.show()
# plotHMFatM(1e14)

def plotHMF3D():
    # X = M in solar Masses
    X = np.linspace(14,16,51)
    Y = np.linspace(0,3,101)
    temp = cosmo.Cosmology()
    Z = np.array([[temp.projected_HMF(np.power(10,M),z)*(np.pi/180)**2 for M in X] for z in Y])
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                           # linewidth=0, antialiased=False, shade=False)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           linewidth=0, cmap = cm.gnuplot)
    plt.title('HMF per deg^2')
    plt.xlabel('log M[solarMasses]')
    plt.ylabel('Redshift')
    plt.show()
# plotHMF3D()