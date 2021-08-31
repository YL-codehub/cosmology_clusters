import matplotlib.pyplot as plt
from scipy.integrate.quadpack import dblquad
import scipy.stats as st
import numpy as np
import math as m
import Cosmological_tools as cosmo
import csv
import time as t
from scipy import interpolate
import scipy.integrate as intg
import scipy.optimize as opt

def readtxt(file):
    X = []
    Y = []
    with open('heavy files/'+file, newline='\n') as csvfile1:
        page1 = csv.reader(csvfile1, quotechar=' ')
        for Row in page1:
            a = Row[0].split()
            if len(a)<=2:
                X.append(float(a[0]))
                Y.append(float(a[1]))
    return(X,Y)


    
def readCoordtxt(file):
    Res = []
    with open(file, newline='\n') as csvfile1:
        page1 = csv.reader(csvfile1, quotechar=' ')
        for Row in page1:
            a = Row[0].split()
            if len(a)<=3:
                Res.append([float(el) for el in a])
    return(np.array(Res))


temp = cosmo.Cosmology()
def fcumul_Dplus(Z,radius):
    '''Repartiion function for radius repartition because of linear growth factor, array in input'''
    integrals = np.ravel([intg.dblquad(lambda z,x: temp.H(z, mode='z') / (x * temp.H(x)) ** 3, 0,np.min([el,radius]),lambda z:0, lambda z:1 / (1 + z))[0]*temp.Ho / temp.constant1 for el in Z])
    norm = intg.dblquad(lambda z,x: temp.H(z, mode='z') / (x * temp.H(x)) ** 3, 0,radius,lambda z:0, lambda z:1 / (1 + z))[0]*temp.Ho / temp.constant1
    return(integrals/norm)



def plotDraw(Catalog):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # radius = 
    # # draw sphere
    # u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    # x = radius*np.cos(u) * np.sin(v)
    # y = radius*np.sin(u) * np.sin(v)
    # z = radius*np.cos(v)

    # alpha controls opacity
    # ax.plot_surface(x, y, z, color="g", alpha=0.3)

    #draw random points:
    for coords in Catalog:
        ax.scatter(*coords,marker = 'o')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_title('Points in a sphere')

    plt.show()

# plotDraw(1000,2)

# Landy and Szaslay estimator for correlation data

def cart2sph(x,y,z):
    '''from 3D cartesian coordinates to spherical ones'''
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # lambda
    az = m.atan2(y,x)                           # phi
    return r, elev, az # rad

def cart2sphA(pts):
    '''list of 3D cartesian coordinates to list of spherical geographic ones'''
    return np.array([cart2sph(x,y,z) for x,y,z in pts])

def sph2cart(a):
    a = np.array(a)
    x = np.array([a[:,0]*np.cos(a[:,1])*np.cos(a[:,2])]).T
    y = np.array([a[:,0]*np.cos(a[:,1])*np.sin(a[:,2])]).T
    z = np.array([a[:,0]*np.sin(a[:,1])]).T
    return np.hstack([x,y,z])

def distanceAB_spherical(A,B):
    return(np.sqrt(abs(A[0]**2+B[0]**2-2*A[0]*B[0]*(np.cos(A[1])*np.cos(B[1])*np.cos(A[2]-B[2])+np.sin(A[1])*np.sin(B[1])))))

def number_Pairs(Data1, Data2, rmin, rmax):
    '''number of pairs Data1(list of spherical coordinates) x Data2 (idem) with a mutual distance betw rmin (strictly) and rmax'''
    counts = 0
    for elt1 in Data1:
        for elt2 in Data2:
            r = distanceAB_spherical(elt1,elt2)
            if r >= rmin and r <= rmax:
                counts+=1
    return(counts)

def CorrelationLS(r,dr,Data, rData):
    '''correlation in bin r-dr/2,r+dr/2 in Data (list of spherical coordinates), with the comparison to rData random mapping.
    Landy and Szaslay estimator formula is used.'''
    N = len(Data)
    rmin, rmax = r-dr/2, r+dr/2
    DD = number_Pairs(Data,Data,rmin,rmax)
    RD = number_Pairs(Data,rData,rmin,rmax)
    RR = number_Pairs(rData,rData,rmin,rmax)
    if RR == 0:
        return(0)
    print(DD,RD,RR)
    return((DD-2*RD+RR)/RR)

def radialcomov_to_redshift(r):
    '''Gives Redshift associated to a given comoving radial distance'''
    Cosmo = cosmo.Cosmology() # usual Omega by default
    func = lambda z : Cosmo.comoving_Distance(z)-r
    dfunc = lambda z : Cosmo.c/Cosmo.H(z,mode = 'z')
    z = opt.newton(func=func,x0=0,fprime=dfunc)
    return(z)

def Comov2z(r):
    step = 0.01
    temp = cosmo.Cosmology()
    zmax = radialcomov_to_redshift(r.max())+0.1
    Z = np.linspace(0,zmax, int((zmax-0)/step)+1)
    temp = cosmo.Cosmology()
    R = [temp.comoving_Distance(z) for z in Z]
    f = interpolate.interp1d(R, Z)
    Redshifts = f(r)
    return(Redshifts)

# tcumul = np.linspace(0.000000001,1,1000)
# fcumul = fcumul_Dplus(tcumul,radius = zmax)
# fcumul_inv = interpolate.interp1d(fcumul,tcumul)
# randomCatalog = random_Ball(radius=zmax,n=rand_n,mode = 'redshifted_sphere',finv = fcumul_inv)

###########################################################
## Verifications :
#####################random ball correlation###############

# Data = readCoordtxt('thresholdcatalog.csv')
# randomData = cart2sphA(random_Ball(np.max(Data[:,0]),len(Data),mode = 'cartesian'))
# # print(CorrelationLS(0.04,0.01,Data,randomData))

# x = np.linspace(0.01,0.07, 70)
# y = [CorrelationLS(el,0.01,Data,randomData) for el in x]
# plt.scatter(x,y,marker = '+')
# plt.xlabel('Radial distance (Mpc)')
# plt.ylabel('Correlation estimator')
# plt.show()

#####################fast distances matrix#################
# import scipy.spatial as sp

# # ref = t.time()
# Catalog = np.loadtxt('thresholdcatalog.txt')
# # n = Catalog.shape[0] #counts number
# zmax = np.max(Catalog[:,0])
# Catalog = sph2cart(Catalog)[:n,:] # au-dessus de 10000 trop lourd
# # plotDraw(Catalog)
# ## random catalog creation, conserving counts density
# rand_radius = 3*zmax
# rand_n = int((rand_radius/zmax)**3)*n
# # print(rand_n) #do not exceed 18 000... for RAM safety
# randomCatalog = random_Ball(radius=rand_radius,n=rand_n)
# # print(t.time()-ref)

# #plot Catalog
# # plotDraw(Catalog)

# # ref = t.time()
# distancesDD = sp.distance_matrix(Catalog,Catalog,2)
# distancesDR = sp.distance_matrix(Catalog,randomCatalog,2)
# distancesRR = sp.distance_matrix(randomCatalog,randomCatalog,2)
# # print(t.time()-ref)

# dr = 10
# # r = np.array([0.5,0.7])
# step = 10
# # zmax = 20
# r = np.array(np.linspace(dr/2,2*zmax,int((2*zmax-dr/2)/step)+1))

# def numberPairs(Dist,r,dr):
#     '''Dist is the distance matrix from two catalogs.
#     r is the np.linspace with all the distances on which bins are centered
#     dr is the width of a distance bin'''
#     return(((Dist>r[:,None,None]-dr/2)&(Dist<r[:,None,None]+dr/2)).sum(axis=(1,2)))


# DD = numberPairs(distancesDD,r,dr)
# DR = numberPairs(distancesDR,r,dr)
# RR = numberPairs(distancesRR,r,dr)
# Xsi = (DD-2*DR+RR)/RR
# plt.plot(r,Xsi)

# #ref
# xref, yref = readtxt('xsi.txt')
# plt.scatter(xref, yref,linewidths=0.05,color = 'yellow')
# plt.legend(['Mine ','Ref'])

# plt.show()
# print(Xsi)


# #####################nbodykit#####################""
def convert_cartesian_to_sky_full_angle(X,Y,Z):
    RA = np.arctan2(X,Z)
    DEC = np.arcsin(Y/np.sqrt(X**2 + Y**2 + Z**2))
    r = np.sqrt(X**2 + Y**2 + Z**2)
    return(r,RA,DEC) #RA = phi, DEC = pi/2-theta

def random_Ball(radius, n, Sig = np.identity(3), mode = 'sphere',finv = None):
    if mode == 'sphere':
        C = st.multivariate_normal(cov = Sig).rvs(size=n)
        U = st.uniform().rvs(size = n)
        U = radius*U**(1/3)
        S = (U*(C.T/np.linalg.norm(C,axis=1))).T
        return S
    if mode == 'redshifted_sphere':
        C = st.multivariate_normal(cov = Sig).rvs(size=n)
        U = st.uniform().rvs(size = n)
        U = finv(U) #cumulative inverse prob function equal to P(D<= z)
        S = (U*(C.T/np.linalg.norm(C,axis=1))).T
        return S
    elif mode == 'cartesian':
        C = st.multivariate_normal(cov = Sig).rvs(size=n)
        maxCoord = np.max(C)
        C = C*radius/maxCoord
        return C
    elif mode == 'test':
        C = np.round((radius*2*(st.uniform().rvs(size = (4*n,3))-0.5))/20)*20
        sphereTruth = (np.linalg.norm(C,axis = 1)<=radius)
        C = C[sphereTruth,:]
        return C[:n,:]

from nbodykit.algorithms.paircount_tpcf import tpcf
from nbodykit.lab import ArrayCatalog
import nbodykit.cosmology.cosmology as cosm

# print('Reading and compiling catalog...')

# Catalog = np.loadtxt('heavy files/BigCatalogTest2.txt')
# # Catalog = np.loadtxt('thresholdcatalog.txt')
# n = Catalog.shape[0] #counts number
# print("Catalog size :",n)
# rmax = np.max(Catalog[:,0])
# data = ArrayCatalog({'RA': Catalog[:,2]*180/np.pi, 'DEC': Catalog[:,1]*180/np.pi, 'Redshift': Catalog[:,0], 'WEIGHT':np.ones(len(Catalog))})

# # # # # # #  random catalog creation
# print('Creating randomized catalog...')

# # ## randomCatalog from uniform
# rand_n = 3*n

# randomCatalog = random_Ball(radius=rmax,n=rand_n,mode = 'test') #ok functionnal
# # print('...and in spherical coordinates...')
# r, RA, DEC = convert_cartesian_to_sky_full_angle(randomCatalog[:,0],randomCatalog[:,1],randomCatalog[:,2])
# random_data = ArrayCatalog({'RA': RA*180/np.pi, 'DEC': DEC*180/np.pi,'Redshift' : r, 'WEIGHT':np.ones(len(r))})

# # randomCatalog = np.loadtxt('heavy files/RANDOMBigcatalogCorrelated.txt')
# # random_data = ArrayCatalog({'RA': randomCatalog[:,2]*180/np.pi, 'DEC': randomCatalog[:,1]*180/np.pi, 'Redshift': randomCatalog[:,0], 'WEIGHT':np.ones(len(randomCatalog))})

# print("Random catalog size :",randomCatalog.shape[0])

# print('Computing Landy and Szaslay estimator...')
# a,b = 15,225
# steps = 10
# bins = np.linspace(a,b,int((b-a)/steps)+1)

# class FakeCosmo(object):
#             def comoving_distance(self, z):
#                 return z
# C = FakeCosmo()

# Xsi = tpcf.SurveyData2PCF(mode='1d',data1=data,randoms1 = random_data, edges = bins,cosmo=C)

# print('Displaying results...')
# plt.subplot(121)
# res = Xsi.corr.data
# r = [el[1] for el in res]
# xsi = [el[0] for el in res]

# ## add errorbars thanks to MONTECARLO preliminary work:
# XsisMC = np.loadtxt('heavy files/XSIsBig.txt').T
# Cov = np.cov(XsisMC)
# # plt.imshow(np.corrcoef(XsisMC))
# # plt.colorbar()
# # plt.show()
# # plt.errorbar(r, xsi, yerr=np.sqrt(np.diag(Cov)),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
# ## http://www.python-simple.com/python-matplotlib/errorBars.php
# plt.scatter(r,xsi,color = 'blue',marker = '+',linewidths = 1.1)

# # np.savetxt('heavy files/binsCorrBig0.txt',r)
# # np.savetxt('heavy files/CorrBig0.txt',xsi)
# # np.savetxt('heavy files/stdCorrBig0.txt',np.sqrt(np.diag(Cov)))

# xref, yref = readtxt('xsi.txt')
# xref,yref = np.array(xref), np.array(yref)
# selection = (xref>=15)&(xref<=r[-1])
# xref,yref = xref[selection], yref[selection]
# yref = 0.2**2*yref
# plt.xlabel('Radial distance (Mpc)')
# plt.ylabel(r'$\xi(r)$')
# plt.plot(xref, yref,color = 'black')
# plt.legend(['Mine','Ref'])

# try:
#     plt.subplot(122)
#     fc = interpolate.interp1d(r,xsi)
#     xsi_2 = fc(xref)
#     plt.plot(xref,np.log(1+xsi_2)/(np.log(1+yref)))
#     plt.xlabel('Radial distance (Mpc)')
#     plt.ylabel(r'$log(1+\xi_{mine}(r))/log(1+\xi_{ref}(r))$')
# except:
#     print('oups')

# plt.show()

# # # # # mpiexec -np 4 python dataCorrelation


# ###################MonteCarlo##################

# a,b = 15,225
# steps = 10
# bins = np.linspace(a,b,int((b-a)/steps)+1)

# class FakeCosmo(object):
#             def comoving_distance(self, z):
#                 return z
# C = FakeCosmo()
# XSIS = []

# for i in range(15,20):
#     print('Iteration: ',i)
#     Catalog = np.loadtxt('heavy files/BigCatalogTest'+str(i)+'.txt')
#     n = Catalog.shape[0] #counts number
#     rmax = np.max(Catalog[:,0])
#     data = ArrayCatalog({'RA': Catalog[:,2]*180/np.pi, 'DEC': Catalog[:,1]*180/np.pi, 'Redshift': Catalog[:,0], 'WEIGHT':np.ones(len(Catalog))})

#     rand_n = 1*n

#     randomCatalog = random_Ball(radius=rmax,n=rand_n,mode = 'test') #ok functionnal
#     r, RA, DEC = convert_cartesian_to_sky_full_angle(randomCatalog[:,0],randomCatalog[:,1],randomCatalog[:,2])
#     random_data = ArrayCatalog({'RA': RA*180/np.pi, 'DEC': DEC*180/np.pi,'Redshift' : r, 'WEIGHT':np.ones(len(r))})


#     Xsi = tpcf.SurveyData2PCF(mode='1d',data1=data,randoms1 = random_data, edges = bins,cosmo=C)

#     res = Xsi.corr.data
#     r = [el[1] for el in res]
#     xsi = [el[0] for el in res]
#     # print(xsi)
#     XSIS.append(xsi)
#     np.savetxt('heavy files/binsCorrBigTest'+str(i)+'.txt',r)
#     np.savetxt('heavy files/CorrBigTest'+str(i)+'.txt',xsi)
    # np.savetxt('heavy files/stdCorrMC'+str(i)+'.txt',np.sqrt(np.diag(Cov)))

# np.savetxt('heavy files/BigTestXSIs.txt',XSIS)


# ########### Big Correlation ###########
# a,b = 15,225
# steps = 10
# bins = np.linspace(a,b,int((b-a)/steps)+1)

# class FakeCosmo(object):
#             def comoving_distance(self, z):
#                 return z
# C = FakeCosmo()

# for i in range(39,40):
#     print('Iteration: ',i)
#     #readCatalog
#     Catalog = np.loadtxt('heavy files/BigCorrelationCatalog'+str(i)+'.txt')
#     n = Catalog.shape[0] #counts number
#     print("Catalog size :",n)
#     rmax = np.max(Catalog[:,0])
#     data = ArrayCatalog({'RA': Catalog[:,2]*180/np.pi, 'DEC': Catalog[:,1]*180/np.pi, 'Redshift': Catalog[:,0], 'WEIGHT':np.ones(len(Catalog))})

#     # randomCatalog
#     randomCatalog = np.loadtxt('heavy files/RANDOMBigCorrelationCatalog'+str(i)+'.txt')
#     random_data = ArrayCatalog({'RA': randomCatalog[:,2]*180/np.pi, 'DEC': randomCatalog[:,1]*180/np.pi, 'Redshift': randomCatalog[:,0], 'WEIGHT':np.ones(len(randomCatalog))})
#     print("Random catalog size :",randomCatalog.shape[0])

#     Xsi = tpcf.SurveyData2PCF(mode='1d',data1=data,randoms1 = random_data, edges = bins,cosmo=C)

#     res = Xsi.corr.data
#     r = [el[1] for el in res]
#     xsi = [el[0] for el in res]

#     np.savetxt('heavy files/CorrbinsBigCorrelation'+str(i)+'.txt',r)
#     np.savetxt('heavy files/CorrBigCorrelation'+str(i)+'.txt',xsi)
    
#     print('-----------')


####simple plot of previous computing
XSIS = []
# XsisMC = np.loadtxt('heavy files/XSIsBig.txt').T
# Cov = np.cov(XsisMC)
# np.savetxt('heavy files/stdCorrBigTest.txt',np.sqrt(np.diag(Cov)))

for i in range(20):
    try:
        r = np.loadtxt('heavy files/binsCorrBigTest'+str(i)+'.txt')
        xsi = np.loadtxt('heavy files/CorrBigTest'+str(i)+'.txt')
        std = np.loadtxt('heavy files/stdCorrBigTest.txt')
        xsi = xsi/(0.2**2)
        plt.errorbar(r, xsi, yerr=std/(0.2**2),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
        plt.scatter(r,xsi,color = 'blue',marker = '+',linewidths = 1.1)


        xref, yref = readtxt('xsi.txt')
        xref,yref = np.array(xref), np.array(yref)
        selection = (xref>=15)&(xref<=r[-1])
        xref,yref = xref[selection], yref[selection]
        # yref = 0.2**2*yref
        plt.xlabel('Radial distance (Mpc)')
        plt.ylabel(r'$\xi(r)$')
        plt.plot(xref, yref,color = 'black')
        plt.legend(['Mine','Ref'])
        plt.show()
    except:
        continue
# plt.show()
np.savetxt('heavy files/BigTestXSIs.txt',XSIS)

