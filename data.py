from astropy.io import fits
from astropy.table import Table
# import astropy.coordinates as coor
import math as m
import numpy as np
import csv
import Cosmological_tools as cosmo
import pickle as pk
import scipy.optimize as opt
import matplotlib.pyplot as plt
# import time as t
from scipy import interpolate

class Catalog:
    def __init__(self, inputfile = None, logM_intervals = 0.1, z_intervals = 0.1):

        self.file = inputfile
        self.entries = []                     # [ {'M':,'z':,'latt':, 'long':}, ...]
        self.logM_intervals = logM_intervals  # for artificial data use only
        self.z_intervals = z_intervals        # for artificial data use only
        self.bins = {}                       # [ {'Mmin':,'zmin':,'latt':, 'long':, 'counts':}, ...] , a bin is [Mmin,Mmin+M_intervals] x [zmin, zmin+z_intervals]
        self.universe = None

    def Generate(self, Minf = 5e14, Msup = 1e16, zinf = 0, zsup = 3, H_0 = 70, Omega_m = 0.3, Omega_r = 0, Omega_v = 0.7, Omega_T = 1, sigma_8 = 0.80, n_s = 0.96):
        '''Artificial counts catalog generator, Ms in solar masses ''' # per deg^2
        a,b = m.log(Minf, 10), m.log(Msup, 10)
        self.universe = cosmo.Cosmology(H_0 = H_0, Omega_m = Omega_m, Omega_r = Omega_r, Omega_v = Omega_v, Omega_T = Omega_T, sigma_8 = sigma_8, n_s = n_s)


        Mmin = np.power(10, np.linspace(a, b, int((b - a) / self.logM_intervals) + 1))
        Mmax = Mmin * 10 ** self.logM_intervals
        zmin = np.linspace(zinf, zsup, int((zsup - zinf) / self.z_intervals) + 1)
        zmax = zmin + self.z_intervals

        # means_poisson = self.universe.expected_Counts(Mmin, Mmax, zmin, zmax, mode = 'superarray')
        # N = np.random.poisson(means_poisson)

        nbin = 1

        for i in range(len(Mmin)): #bin [Mm, Mm+Minterval], but exponential dependance
            for j in range(len(zmin)): # bin [zm, zm+z_interval]
                ## Mean counts in the bin thanks to theory
                mean_poisson = self.universe.expected_Counts(Mmin[i],Mmax[i],zmin[j],zmax[j]) # per srad
                N = np.random.poisson(mean_poisson)
        #         ## Encode data :
                self.bins[nbin] = {'Mmin':Mmin[i], 'zmin':zmin[j], 'latt':None, 'long':None, 'counts':N}
                # print(self.bins[nbin])
                nbin +=1

        # for i in range(len(Mmin)): #bin [Mm, Mm+Minterval], but exponential dependance
        #     for j in range(len(zmin)): # bin [zm, zm+z_interval]
        # #         ## Encode data :
        #         self.bins[nbin] = {'Mmin':Mmin[i], 'zmin':zmin[j], 'latt':None, 'long':None, 'counts':N[i,j]}
        #         # print(self.bins[nbin])
        #         nbin +=1

    def save_bins(self,name):
        a_file = open(name, "wb")
        pk.dump(self.bins, a_file)
        a_file.close()

    def read_bins(self,name):
        a_file = open(name, "rb")
        self.bins = pk.load(a_file)

    def plotLikelihood1D(self,S,H_0 = 70, Omega_m_ini = 0.5, Omega_r = 0, Omega_v = 0.7, Omega_T = 1, sigma_8_ini = 0.50, n_s = 0.96):
        '''Univariate likelihood maximum gives cosmological parameters Omega_m and sigma_8'''
        self.universe = cosmo.Cosmology(H_0=H_0, Omega_m=Omega_m_ini, Omega_r=Omega_r, Omega_v=Omega_v, Omega_T=Omega_T,sigma_8=sigma_8_ini, n_s=n_s)
        # Mmin = np.ravel([self.bins[k]['Mmin'] for k in self.bins.keys()])
        # Zmin = np.ravel([self.bins[k]['zmin'] for k in self.bins.keys()])
        # Counts = np.ravel([self.bins[k]['counts'] for k in self.bins.keys()])
        def loglikelihood(parameters):
            print(parameters)
            res = 0
            self.universe.Om = parameters[0]
            self.universe.sigma8 = parameters[1]
            self.universe.update()
            for k in self.bins.keys():
                mean_poisson = self.universe.expected_Counts(self.bins[k]['Mmin'],
                                                             self.bins[k]['Mmin'] * 10**self.logM_intervals,
                                                             self.bins[k]['zmin'],
                                                             self.bins[k]['zmin'] + self.z_intervals)
                if mean_poisson != 0:
                    res += -mean_poisson + self.bins[k]['counts'] * m.log(mean_poisson)  # take the opposite to take the minimum


            # res += -mean_poisson + self.bins[k]['counts'] * np.log(mean_poisson)
            # means_poisson = self.universe.expected_Counts(Mmin, Mmin * 10 ** self.logM_intervals,Zmin, Zmin + self.z_intervals, mode='superarray')
            # res = -means_poisson + Counts * np.log(means_poisson)  # take the opposite to take the minimum
            return np.sum(res)

        Y = np.ravel([loglikelihood([0.3,s]) for s in S]) #sigma 8 varies
        # Y = np.ravel([loglikelihood([Om, 0.8]) for Om in S]) # Omega_m varies
        # Y = np.exp(Y)
        Y = Y-Y.min()
        # Y = Y/Y.sum()
        Y = np.exp(Y-Y.max()) #normalize to get computable exponential, argmax unchanged
        Y = Y/Y.sum() #normalize exp
        t = np.linspace(0, Y.max(), 1000)
        integral = ((Y >= t[:, None, None]) * Y).sum(axis = (1,2))
        # t = np.linspace(0, m.exp(Y.max()), 1000)
        # integral = np.exp(Y >= t[:, None, None] * Y).sum(axis=(1, 2))
        f = interpolate.interp1d(integral, t)
        t_contours = f(np.array([0.95, 0.68]))
        fig, ax = plt.subplots()
        ax.plot(S,Y)
        plt.fill_between(S, 0, Y, where= Y >= t_contours[0], color='r', alpha=.3)
        plt.fill_between(S, 0, Y, where= Y >= t_contours[1], color='b', alpha=.3)
        # plt.plot(S,Y)
        plt.ylabel('likelihood')
        plt.xlabel('parameter')
        plt.show()

    def plotLikelihood(self,O,S,H_0 = 70, Omega_m_ini = 0.5, Omega_r = 0, Omega_v = 0.7, Omega_T = 1, sigma_8_ini = 0.50, n_s = 0.96):
        '''Univariate likelihood maximum gives cosmological parameters Omega_m and sigma_8'''
        self.universe = cosmo.Cosmology(H_0=H_0, Omega_m=Omega_m_ini, Omega_r=Omega_r, Omega_v=1-Omega_m_ini, Omega_T=Omega_T,sigma_8=sigma_8_ini, n_s=n_s)
        # Mmin = np.ravel([self.bins[k]['Mmin'] for k in self.bins.keys()])
        # Zmin = np.ravel([self.bins[k]['zmin'] for k in self.bins.keys()])
        # Counts = np.ravel([self.bins[k]['counts'] for k in self.bins.keys()])
        def loglikelihood(parameters):
            print(parameters)
            res = 0
            self.universe = cosmo.Cosmology(H_0=H_0, Omega_m=parameters[0], Omega_r=Omega_r, Omega_v=1-parameters[0], sigma_8=parameters[1], n_s=n_s)
            # self.universe.Om = parameters[0]
            # self.universe.sigma8 = parameters[1]
            # self.universe.update()
            for k in self.bins.keys():
                mean_poisson = self.universe.expected_Counts(self.bins[k]['Mmin'], self.bins[k]['Mmin']*10**self.logM_intervals, self.bins[k]['zmin'],self.bins[k]['zmin'] + self.z_intervals)
                res += -mean_poisson+self.bins[k]['counts']*m.log(mean_poisson) # take the opposite to take the minimum
            # res += -mean_poisson + self.bins[k]['counts']*np.log(mean_poisson)
            # means_poisson = self.universe.expected_Counts(Mmin, Mmin * 10 ** self.logM_intervals,Zmin, Zmin + self.z_intervals, mode='superarray')
            # res = -means_poisson + Counts * np.log(means_poisson)  # take the opposite to take the minimum
            return res

        Z = np.array([[loglikelihood([o,s]) for s in S] for o in O])
        np.savetxt('countsresultZ',Z)
        np.savetxt('countsresultO',O)
        np.savetxt('countsresultS',S)

        a = np.argmax(Z)
        print('Omega_m :', O[a % len(S)])
        print('Sigma :', S[a // len(S)])
        #
        # X, Y = np.meshgrid(O, S)
        # Z = np.exp(Z - Z.max())
        # Z = Z / Z.sum()
        # #
        # t = np.linspace(0, Z.max(), 1000)
        # integral = ((Z >= t[:, None, None]) * Z).sum(axis=(1, 2))
        # f = interpolate.interp1d(integral, t)
        # t_contours = f(np.array([0.95, 0.68]))
        # plt.contour(X, Y, Z, t_contours)
        # plt.colorbar()
        # plt.scatter(O[a % len(S)], S[a // len(S)])
        # plt.show()

    def refine(self,H_0 = 70, Omega_m_ini = 0.5, Omega_r = 0, Omega_v = 0.7, Omega_T = 1, sigma_8_ini = 0.50, n_s = 0.96):
        '''Univariate likelihood maximum gives cosmological parameters Omega_m and sigma_8'''
        self.universe = cosmo.Cosmology(H_0=H_0, Omega_m=Omega_m_ini, Omega_r=Omega_r, Omega_v=Omega_v, Omega_T=Omega_T,sigma_8=sigma_8_ini, n_s=n_s)
        def mloglikelihood(parameters):
            res = 0
            # self.universe = cosmo.Cosmology(H_0=H_0, Omega_m=parameters[0], Omega_r=Omega_r, Omega_v=Omega_v,
            #                                 Omega_T=Omega_T, sigma_8=parameters[1], n_s=n_s)
            self.universe.Om = parameters[0]
            self.universe.sigma8 = parameters[1]
            self.universe.update()
            #oups je crois qu'il faut aussi relancer le calcul d'As ducoup...
            for k in self.bins.keys(): #pas bon
                mean_poisson = self.universe.expected_Counts(self.bins[k]['Mmin'], self.bins[k]['Mmin']*(1+ m.exp( self.logM_intervals)),
                                                        self.bins[k]['zmin'],self.bins[k]['zmin'] + self.z_intervals, (m.pi / 180) ** 2)
                res -= -mean_poisson+self.bins[k]['counts']*m.log(mean_poisson) # take the opposite to take the minimum
            print(parameters)
            return res

        x = opt.minimize(mloglikelihood, [Omega_m_ini,sigma_8_ini],options={ 'disp': True}, bounds = ((0,1),(0,2)))
        return x

# #
# temp = Catalog()
# temp.Generate()
# temp.save_bins('test')
# # please take the same rad2
# temp = Catalog()
# temp.read_bins('test')
# temp.plotLikelihood(np.linspace(0.25,0.35,40),np.linspace(0.75,0.85,40))
# temp.plotLikelihood1D(np.linspace(0.5,1,20)) #sigma8
# temp.plotLikelihood1D(np.linspace(0.2,0.4,10)) #Omega M

# # debug contours
O = np.loadtxt('countsresultO')
S = np.loadtxt('countsresultS')
Z = np.loadtxt('countsresultZ')


a = np.argmax(Z)
## Check
print('Omega_m :',O[a%len(S)])
print('Sigma :',S[a//len(S)])
X, Y = np.meshgrid(O,S)
Z = np.exp(Z-Z.max())
Z = Z/Z.sum()
#
t = np.linspace(0, Z.max(), 1000)
integral = ((Z >= t[:, None, None]) * Z).sum(axis=(1, 2))
f = interpolate.interp1d(integral,t)
t_contours = f(np.array([0.95,0.680]))
cs = plt.contour(X,Y,Z,t_contours, colors = ['red','blue'],alpha = 0.5)
plt.scatter(O[a%len(S)],S[a//len(S)],marker= '+')
plt.grid()
plt.title('Constraints on cosmological parameters thanks to Poisson max likelihood.\n'
          ' Contours of confidence at 95% (red) and 68% (blue).')
plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$\sigma_8$')
# plt.show()

# ## MonteCarlo :
# temp = Catalog()
# for i in range(10):
#     temp.bins = {}
#     temp.Generate()
#     temp.plotLikelihood(np.linspace(0.2, 0.4, 20), np.linspace(0.7, 0.9, 20))

#REsults (20x20 maillage) :
Omega_m = [0.2947368421052632, 0.2947368421052632, 0.30526315789473685, 0.2947368421052632, 0.30526315789473685, 0.2947368421052632, 0.30526315789473685, 0.30526315789473685, 0.2947368421052632, 0.30526315789473685]
Sigma = [0.8052631578947368, 0.8157894736842105, 0.7947368421052632, 0.8157894736842105, 0.7842105263157895, 0.8157894736842105, 0.7947368421052632, 0.7947368421052632, 0.8157894736842105, 0.7947368421052632]
plt.scatter(Omega_m,Sigma,linewidths=0.2,marker = '+')
plt.show()
####################################"""
def readFits(file):
    with fits.open(file) as hdu:
        asn_table = Table(hdu[1].data)
    return (np.array(asn_table['GLON', 'GLAT', 'REDSHIFT']))


# print(readFits('HFI_PCCS_SZ-union_R2.08.fits'))


def readtxt(file):
    X = []
    Y = []
    with open(file, newline='\n') as csvfile1:
        page1 = csv.reader(csvfile1, quotechar=' ')
        for Row in page1:
            a = Row[0].split()
            if len(a)<=2:
                X.append(float(a[0]))
                Y.append(float(a[1]))
    return(X,Y)


# print(readtxt('covolume.txt'))
########"""""
# c = coor.SkyCoord(x=1, y=2, z=3, unit='kpc', representation_type='cartesian')
# [a,b,c] = coor.cartesian_to_spherical(1,1,1)
# print(a,b,c)
#
#
# print(Carttospher([1,1,1]))
#
#
# def cart2sph(x,y,z):
#     '''from 3D cartesian coordinates to spherical ones'''
#     XsqPlusYsq = x**2 + y**2
#     r = m.sqrt(XsqPlusYsq + z**2)               # r
#     elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # lambda
#     az = m.atan2(y,x)                           # phi
#     return r, elev, az # rad
#
# def cart2sphA(pts):
#     '''list of 3D cartesian coordinates to list of spherical geographic ones'''
#     return np.array([cart2sph(x,y,z) for x,y,z in pts])
# print(cart2sph(1,1,1))
# print(cart2sphA([[1,1,1],[1,2,3]]))