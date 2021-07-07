# (another possibility : use astropy.cosmolgy.FLRW package...)

import math as m
import scipy.integrate as intg
import matplotlib.pyplot as plt
import numpy as np
import csv

## Definition of a cosmology
class Cosmology:
    def __init__(self, H_0 = 70, Omega_m = 0.3, Omega_r = 0, Omega_v = 0.7, Omega_T = 1, sigma_8 = 0.80, n_s = 0.96, k = 0):

        # Cosmological parameters
        self.Ho = H_0         # Hubble constant today (units = km.s^-1.Mpc^-1)
        self.Om = Omega_m     # Normalized density parameter for matter (units = 1)
        self.Or = Omega_r     # Normalized density parameter for radiations (units = 1)
        self.Ov = Omega_v     # Normalized density parameter for vacuum (units = 1)
        self.OT = Omega_T     # Normalized total density parameter (units = 1)
        self.sigma8 = sigma_8 # RMS density parameter at 8 h^-1 Mpc (units = 1)
        self.ns = n_s         # Inflation exponent in Power spectrum law (units = 1)
        self.k = k            # Curvature -1, 0 ( <=> Omega_T = 1) or 1 (units = 1)

        # Physical constants
        self.Mpc = 3.261563e6  # al
        self.year = 3600*365.25*24 # 1 year in sec
        self.c = 2.99792458e5 # (km.s^-1)
        self.solarMass = 1.98847e30 # (kg)
        self.Mpc_m = self.Mpc*self.c*1e3*self.year
        self.G = 6.67384e-11*1e-6*self.solarMass/(self.Mpc_m)# (Mpc^1.km^2.solarMass^-1.s^-3)
        self.Ho_sec = self.Ho/(self.Mpc*self.c*self.year) # Hubble constant today (units = s^-1)
        self.h = self.Ho/100

        # Others
        if self.k == 0:
            self.R0 = 1
        else:
            self.R0 = self.c/(self.Ho*m.sqrt(abs(1-self.OT))) # current radius (units = Mpc)
        self.critical_density0 = 3*self.Ho**2/(8*m.pi*self.G) # Critical density today (units = SolarMass^1.Mpc^-3)
        self.density0 = (self.Om+self.Or+self.Ov)*self.critical_density0  # Mean density of the universe today (units = SolarMass^1.Mpc^-3)
        self.As = self.sigma8 ** 2 * (2 * m.pi) ** 3 / (4*m.pi*intg.quad(lambda k : k ** self.ns * self.transfer_Function_BKKS(k) ** 2 * abs(self.window(k * 8/self.h))** 2 * k ** 2, 0, m.inf)[0])  # Power spectrum law's normalization

        #storing
        self.constant1 = intg.quad(lambda x : 1/(x*self.H(x))**3, 0, 1)[0]

    ## Properties
    def update(self):
        '''update properties, has to be called when attributes change'''
        self.Ho_sec = self.Ho / (self.Mpc * self.c * self.year)  # Hubble constant today (units = s^-1)
        self.h = self.Ho / 100
        if self.k == 0:
            self.R0 = 1
        else:
            self.R0 = self.c/(self.Ho*m.sqrt(abs(1-self.OT))) # current radius (units = Mpc)
        self.critical_density0 = 3*self.Ho**2/(8*m.pi*self.G) # Critical density today (units = SolarMass^1.Mpc^-3)
        self.density0 = (self.Om+self.Or+self.Ov)*self.critical_density0  # Mean density of the universe today (units = SolarMass^1.Mpc^-3)
        self.As = self.sigma8 ** 2 * (2 * m.pi) ** 3 / (4*m.pi*intg.quad(lambda k : k ** self.ns * self.transfer_Function_BKKS(k) ** 2 * abs(self.window(k * 8/self.h))** 2 * k ** 2, 0, m.inf)[0])  # Power spectrum law's normalization
        self.constant1 = intg.quad(lambda x: 1 / (x * self.H(x)) ** 3, 0, 1)[0]
    # @property
    # def R0(self): # current radius (units = Mpc)
    #     if self.k == 0:
    #         return 1
    #     else:
    #         return self.c / (self.Ho * m.sqrt(abs(1 - self.OT)))
    #
    # @property  # Critical density today (units = SolarMass^1.Mpc^-3)
    # def critical_density0(self):
    #     return 3 * self.Ho ** 2 / (8 * m.pi * self.G)
    #
    # @property
    # def density0(self):# Mean density of the universe today (units = SolarMass^1.Mpc^-3)
    #     return (self.Om + self.Or + self.Ov) * self.critical_density0
    #
    # @property
    # def As(self):  # Power spectrum law's normalization
    #     return self.sigma8 ** 2 * (2 * m.pi) ** 3 / (4*m.pi*intg.quad(lambda k : k ** self.ns * self.transfer_Function_BKKS(k) ** 2 * abs(self.window(k * 8/self.h))** 2 * k ** 2, 0, m.inf)[0])

    ## Tools
    def S(self,x):
        '''S function parametrizing the FLRW metric depending on the curvature'''
        if self.k == 1:
            return(m.sin(x))
        elif self.k == 0:
            return(x)
        elif self.k == -1:
            return(m.sin(x)) # erreur

    def derivative(self,func,x, dx = 1e-6): #peut-être utiliser dérivation d'un librairie
        '''1st derivative of a given function with central difference approximation with dx gap'''
        return((func(x+dx)-func(x))/dx)

## Different indicative calculus
    def H(self,x,mode = 'a'):
        ''' Hubble parameter value (units = km.s^-1.Mpc^-1) at a given redshift ('z' mode) or at a given expansion rate ('a' mode) '''
        if mode == 'a':
            return(self.Ho*m.sqrt(self.Om*x**-3 +self.Or*x**-4 + self.Ov + (1-self.OT)*x**-2))
        elif mode == 'z':
            return(self.Ho*m.sqrt(self.Om*(1+x)**3 +self.Or*(1+x)**4 + self.Ov + (1-self.OT)*(1+x)**2))

    def age(self, a):
        ''' Age (units = Gyr) of the universe at a given expansion parameter a'''
        return( intg.quad(lambda x : 1/(x*self.H(x)), 0, a)[0]*(self.Mpc*self.c)*1e-9 )
    def density(self,z):
        '''density (units = SolarMass^1.Mpc^-3) of the universe at a given redshift z'''
        return(self.critical_density0*(self.Om*(1+z)**3+self.Or*(1+z)**4+self.Ov))

## Computing Distances
    def comoving_Distance(self, z):
        ''' Comoving distance (Mpc) or radial photon distance from a given redshift z to now'''
        return( intg.quad(lambda x : 1/(x**2*self.H(x)), 1/(1+z), 1)[0]*self.c )

    def light_travel_time_Distance(self, z):
        ''' Light travel time Distance (Mpc) or proper photon distance from a given redshift z to now'''
        return( intg.quad(lambda x : 1/(x*self.H(x)), 1/(1+z), 1)[0]*self.c )

    def transverse_comoving_Distance(self, z):
        ''' Transverse comoving distance (Mpc) or radial photon distance from a given redshift z to now'''
        return( self.R0*self.S(self.comoving_Distance(z)/self.R0) )

    def angular_diameter_Distance(self, z):
        ''' Angular Diameter distance (Mpc) or curved photon distance from a given redshift z to now'''
        return(self.transverse_comoving_Distance(z)/(1+z))

    def luminosity_Distance(self, z):
        ''' Luminosity distance (Mpc) or equivalent distance associated to a given flux of energy coming from a blackbody at a given redshift z to now'''
        return(self.transverse_comoving_Distance(z)*(1+z))

    def comoving_Volume(self,z):
        '''Comoving volume (units = Mpc^3)'''
        return(4*m.pi*self.comoving_Distance(z)**3/3)

    def differential_comoving_Volume(self,z):
        '''Differential comoving volume (units = Mpc^3) necessary to express HMF in a portion of the sky only and of the redshifts only'''
        return(self.c*(1+z)**2*self.angular_diameter_Distance(z)**2/self.H(z,mode='z'))

## Tools for Clusters Counting
    def Dplus(self,z): #remarque : cf thèse emmanuel artis, Halo model ---> D(a) ~ a
        '''Linear gross factor (units = 1) at a given redshift z, associated to density-type expansions like density contrast'''
        return( (self.H(z,mode = 'z')/self.Ho)*intg.quad(lambda x : 1/(x*self.H(x))**3, 0, 1/(1+z))[0]/self.constant1)

    def transfer_Function_BKKS(self,k,theta =1):
        '''Transfer function at wavenumber k (units Mpc^-1)'''
        q = k*theta**0.5/(self.Om*self.h**2) #Mpc
        if k == 0:
            return(1)
        else:
            return((m.log(1+2.34*q)/(2.34*q))*(1+3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25))

    def window(self,y):
        '''Window function in Fourier space, the product with which allows to get rid of low values of radius or mass'''
        try:
            if y <1e-7:
                return(1)
            elif y == m.inf:
                return(0)
            else:
                return(3*(m.sin(y)/y-m.cos(y))/y**2)
        except ValueError:
            print(y)

    def initial_Power_Spectrum_BKKS(self,k):
        '''spatial part of Power spectrum (units = Mpc^3) at wavenumber k (units = Mpc^-1)'''
        return(self.As*k**self.ns*self.transfer_Function_BKKS(k)**2)

    def initial_sigma(self,M):
        '''Initial RMS density fluctuation (units = 1) over a given mass M (units = solar Mass)'''
        ### Attention c'est bien la densité de matière qu'il nous faut
        R = (3*M/(4*m.pi*self.Om*self.critical_density0))**(1/3) # (units = Mpc)
        # return(m.sqrt(4*m.pi*intg.quad(lambda k : self.initial_Power_Spectrum_BKKS(k) * self.window(k * R)** 2 * k ** 2 ,0, m.inf,limit = 100)[0] / ((2 * m.pi) ** 3)))
        # return (m.sqrt(4 * m.pi * intg.fixed_quad(lambda u: self.initial_Power_Spectrum_BKKS(m.exp(u[0])) * self.window(m.exp(u[0]) * R) ** 2 * m.exp(3*u[0]), -100, 100)[0] / ((2 * m.pi) ** 3)))
        return (m.sqrt(4 * m.pi *
                       intg.quad(lambda k: self.initial_Power_Spectrum_BKKS(k) * self.window(k * R) ** 2 * k ** 2, 0,
                                 m.inf,epsabs = 1)[0] / ((2 * m.pi) ** 3)))

    # def initial_sigmaref(self, M):
    #     '''Initial RMS density fluctuation (units = 1) over a given mass M (units = solar Mass)'''
    #     X, Yref = rd.readtxt('sigmaM.txt')
    #     return(Yref[X.index(round(M*1e-15,7))])

    def Press_Schechter_multiplicity_Function(self,M,z,delta_c = 1.686):
        '''Gaussian multiplicity function used in the Press-Schechter HMF formulation  at mass M (units = solarMass) and redshift z'''
        sigma = self.Dplus(z)*self.initial_sigma(M)
        return(m.sqrt(2/m.pi)*(delta_c/sigma)*m.exp(-0.5*(delta_c/sigma)**2))

    def Tinker_multiplicity_Function(self,M,z):
        sigma = self.Dplus(z) * self.initial_sigma(M)
        a = lambda z: 1.47*(1+z)**(-0.06)
        delta = 200
        alpha = m.exp(-(0.75/m.log(delta/75))**1.2)
        b = lambda z: 0.257*(1+z)**(-alpha)
        c = 1.19
        A = lambda z: 0.186*(1+z)**(-0.14)
        return(A(z)*((b(z)/sigma)**(a(z))+1)*m.exp(-c/sigma**2))

    # def dlnsig(self, M, multiplicity='PS'):  # density0 today ?
    #     '''Halo Mass Function, dn/dlnM'''
    #     # der = abs(self.derivative(lambda mass: m.log(self.initial_sigma(mass)), M,dx=1e15 * 1e-8))  # revenir à du logarithmique, avec pas de mass en log
    #     der = abs(self.derivative(lambda lnmass: m.log(self.initial_sigma(m.exp(lnmass))), m.log(M), dx=m.log(1e15) * 1e-8)) # revenir à du logarithmique, avec pas de mass en log
    #     return der

    # def dlnsigref(self, M):
    #     '''Initial RMS density fluctuation (units = 1) over a given mass M (units = solar Mass)'''
    #     X, Yref = rd.readtxt('dlnsigmadlnM.txt')
    #     return(Yref[X.index(round(M*1e-15,7))])

    def HMF(self,M,z,multiplicity = 'PS'): #density0 today ?
        '''Halo Mass Function, dn/dlnM'''
        # der = abs(self.derivative(lambda logmass : m.log(self.initial_sigma(m.exp(logmass))), m.log(M),dx = m.log(1e15)*1e-8))
        # der/=M #too slow : why ?
        der = abs(self.derivative(lambda mass: m.log(self.initial_sigma(mass)), M, dx= 1e15 * 1e-8))

        if multiplicity == 'PS':
            return(self.critical_density0*self.Om * der*self.Press_Schechter_multiplicity_Function(M,z))
        elif multiplicity == 'T':
            return (self.critical_density0*self.Om * der * self.Tinker_multiplicity_Function(M, z))

    def projected_HMF(self,M,z, multiplicity = 'PS'):
        '''dN/(dz dOmega dlnM) = Number of objects per unit of projected area on the sky and redshift (units = srad^-2) '''
        return(self.HMF(M,z,multiplicity)*self.differential_comoving_Volume(z))

    def expected_Counts(self,Mmin,Mmax,zmin,zmax, rad2):
        '''Expected counts in a rad2 (units = rad^2) portion of the sky, given the Halo theory and so given the HMF density.'''
        # return(rad2 * intg.dblquad(lambda lnM,z : self.projected_HMF(m.exp(lnM),z), m.log(Mmin), m.log(Mmax), lambda x : zmin, lambda x : zmax)[0]) #dlnM = dM/M
        return (rad2 * intg.dblquad(lambda lnM, z: self.projected_HMF(m.exp(lnM), z), m.log(Mmin), m.log(Mmax), lambda x: zmin,
                             lambda x: zmax)[0])  # dlnM = dM/M


temp = Cosmology(Omega_m=0.3,Omega_v = 0.7)
# #
# print(temp.expected_Counts(5e14,1e16,0,5,(m.pi/180)**2)) # 1 srad

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

def checkplot(temp,file):
    '''temp = cosmology object'''
    X, Yref = readtxt(file)
    # plt.plot(X,Yref, color = 'red')
    plt.loglog(X,Yref, color = 'red')
    #
    # # Angular diameter distance
    # Y = [temp.angular_diameter_Distance(x) for x in X]
    # plt.xlabel('Redshift z (units = 1)')
    # plt.ylabel('Angular Diameter Distance (units = Mpc)')

    # # # Growth factor
    # Y = [temp.Dplus(x) for x in X]
    # plt.xlabel('Redshift z (units = 1)')
    # plt.ylabel('Growth factor (units = 1)')

    # # # Differential comoving volume
    # Y = [temp.differential_comoving_Volume(x) for x in X]
    # plt.xlabel('Redshift z (units = 1)')
    # plt.ylabel('Differential Comoving Volume (units = Mpc^3)')

    # # # # Initial power spectrum
    # Y = [temp.initial_Power_Spectrum_BKKS(x) for x in X]
    # plt.xlabel('Wave number (units = Mpc^-1)')
    # plt.ylabel('Initial Power Spectrum P(k) (units = Mpc^3)')

    # # # RMS density fluctuation
    # Y = [temp.initial_sigma(x*1e15) for x in X]
    # plt.xlabel('Mass (units = 1e15 SolarMass)')
    # plt.ylabel('RMS density fluctuation (units = 1)')

    # # # dlnsig/dlnM
    # Y = [temp.dlnsig(x*1e15,1) for x in X]
    # plt.xlabel('Mass (units = 1e15 SolarMass)')
    # plt.ylabel('dln sigma /dlnM(M,z=1) for')

    # # Projected HMF (Press-Schechter)
    Y = [temp.projected_HMF(x*1e15,1)*(m.pi/180)**2 for x in X]
    plt.xlabel('Mass (units = 1e15 SolarMass)')
    plt.ylabel('dN/dz/dlnM(M,z=1) for 1 square degree')
    # #
    # # # Projected HMF (Tinker)
    # Y = [temp.projected_HMF(x*1e15,1,multiplicity='T')*(m.pi/180)**2 for x in X]
    # plt.xlabel('Mass (units = 1e15 SolarMass)')
    # plt.ylabel('dN/dz/dlnM(M,z=1) for 1 square degree')

    ######### plot :
    plt.grid()
    # plt.plot(X,Y, linestyle = '--',color = 'blue')
    plt.loglog(X,Y, linestyle = '--',color = 'blue')
    # plt.loglog(X,(np.ravel(Y)/np.ravel(Yref))/np.ravel(Yref))
    plt.semilogx()
    # plt.legend(['Ref','Mine'])
    plt.title('H0 = 70, OmegaM=0.7, OmegaV=0.3, sigma = 0.8, n_s = 0.96')
    plt.show()
#
# temp = Cosmology()
# checkplot(temp,'hmfM_PS.txt')
# #
# import time as t
# Time = t.time()
print(temp.expected_Counts(1e14,10**(14.01),1,1.01,(m.pi/180)**2)) # 1 srad
# print(t.time()-Time)

