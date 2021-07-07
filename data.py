from astropy.io import fits
from astropy.table import Table
# import astropy.coordinates as coor
import math as m
import numpy as np
import csv
import random as rd
import Cosmological_tools as cosmo
import pickle as pk
import scipy.optimize as opt

class Catalog:
    def __init__(self, inputfile = None, logM_intervals = 0.01, z_intervals = 0.01):

        self.file = inputfile
        self.entries = []                     # [ {'M':,'z':,'latt':, 'long':}, ...]
        self.logM_intervals = logM_intervals  # for artificial data use only
        self.z_intervals = z_intervals        # for artificial data use only
        self.bins = {}                       # [ {'Mmin':,'zmin':,'latt':, 'long':, 'counts':}, ...] , a bin is [Mmin,Mmin+M_intervals] x [zmin, zmin+z_intervals]
        self.universe = None

    def Generate(self, Minf = 5e14, Msup = 1e16, zinf =0.95, zsup = 0.96, H_0 = 70, Omega_m = 0.3, Omega_r = 0, Omega_v = 0.7, Omega_T = 1, sigma_8 = 0.80, n_s = 0.96):
        '''Artificial counts catalog generator, Ms in solar masses ''' # per deg^2
        Ms = np.exp(np.linspace(m.log(Minf),m.log(Msup),int((m.log(Msup)-m.log(Minf))/self.logM_intervals)+1))
        Zs = np.linspace(zinf,zsup,int((zsup-zinf)/self.z_intervals)+1)
        self.universe = cosmo.Cosmology(H_0 = H_0, Omega_m = Omega_m, Omega_r = Omega_r, Omega_v = Omega_v, Omega_T = Omega_T, sigma_8 = sigma_8, n_s = n_s)
        nbin = 1
        for Mm in Ms: #bin [Mm, Mm+Minterval], but exponential dependance
            for zm in Zs: # bin [zm, zm+z_interval]
                ## Mean counts in the bin thanks to theory
                mean_poisson = self.universe.expected_Counts(Mm,Mm*(1+m.exp(self.logM_intervals)),zm,zm+self.z_intervals,(m.pi/180)**2) # per deg^2
                print(mean_poisson)
                N = np.random.poisson(mean_poisson)
                ## Adding some noise (1% std and gaussian)
                # N *= rd.gauss(1,0.01)
                ## Encode data
                print({'Mmin':Mm, 'zmin':zm, 'latt':None, 'long':None, 'counts':N})
                self.bins[nbin] = {'Mmin':Mm, 'zmin':zm, 'latt':None, 'long':None, 'counts':N}
                nbin +=1

    def save_bins(self,name):
        a_file = open(name, "wb")
        pk.dump(self.bins, a_file)
        a_file.close()

    def read_bins(self,name):
        a_file = open(name, "rb")
        self.bins = pk.load(a_file)

    def refine(self,H_0 = 70, Omega_m_ini = 0.5, Omega_r = 0, Omega_v = 0.7, Omega_T = 1, sigma_8_ini = 0.50, n_s = 0.96):
        '''Univariate likelihood maximum gives cosmological parameters Omega_m and sigma_8'''
        self.universe = cosmo.Cosmology(H_0=H_0, Omega_m=Omega_m_ini, Omega_r=Omega_r, Omega_v=Omega_v, Omega_T=Omega_T,sigma_8=sigma_8_ini, n_s=n_s)
        def loglikelihood(parameters):
            res = 0
            self.universe.Om = parameters[0]
            self.universe.sigma8 = parameters[1]
            self.universe.update()
            #oups je crois qu'il faut aussi relancer le calcul d'As ducoup...
            for k in self.bins.keys():
                mean_poisson = self.universe.expected_Counts(self.bins[k]['Mmin'], self.bins[k]['Mmin']*(1+ m.exp( self.logM_intervals)),
                                                        self.bins[k]['zmin'],self.bins[k]['zmin'] + self.z_intervals, (m.pi / 180) ** 2)
                res -= -mean_poisson+self.bins[k]['counts']*m.log(mean_poisson) # take the opposite to take the minimum
            print(parameters)
            return res

        x = opt.minimize(loglikelihood, [Omega_m_ini,sigma_8_ini],options={ 'disp': True}, bounds = ((0,1),(0,2)))
        return x

#
temp = Catalog()
temp.Generate()
temp.save_bins('test')
#
# temp = Catalog()
# temp.read_bins('test')
# print(temp.refine())

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
