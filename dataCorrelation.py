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

#choose a mode :

mode = None
# mode = 'perso' ## then my version, MC implementation
# mode = 'nbodykit'
# mode = 'MC'

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

def integralXsi(R,univ,a = 1e-7, b= 1e3,n = 100000):
    K = np.array([np.linspace(a, b, n)])
    dK = (b - a) / n
    X = np.array([R]).T
    F = univ.initial_Power_Spectrum_BKKS(K, mode='np') * np.sin(K * X) / (K * X) * K ** 2 / (2 * np.pi ** 2)
    xsi_model = np.sum(F, axis=1) * dK
    return(xsi_model)
    
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

    #draw random points:
    for coords in Catalog:
        ax.scatter(*coords,marker = 'o')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_title('Points in a sphere')

    plt.show()

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

#####################fast distances matrix#################
plot = True
save = False

if mode == 'perso':
    import scipy.spatial as sp
    XSIS = []
    # ref = t.time()
    for i in range(1):
        print('Iteration :',str(i))
        Catalog = np.loadtxt('heavy files/BigCatalogMCContinuous'+str(i)+'.txt')
        Catalog = np.nan_to_num(Catalog)
        # Catalog = np.loadtxt('thresholdcatalog.txt')
        n = Catalog.shape[0] #counts number
        print("Catalog size :",n)
        rmax = np.max(Catalog[:,0])
        Catalog = sph2cart(Catalog)[:n,:] # au-dessus de 10000 trop lourd
        # plotDraw(Catalog)
        ## random catalog creation, conserving counts density
        # rand_radius = 3*zmax
        # rand_n = int((rand_radius/zmax)**3)*n
        rand_n = n
        # print(rand_n) #do not exceed 18 000... for RAM safety
        randomCatalog = random_Ball(radius=rmax,n=rand_n)
        # print(t.time()-ref)

        #plot Catalog
        # plotDraw(Catalog)

        # ref = t.time()
        distancesDD = sp.distance_matrix(Catalog,Catalog,2)
        distancesDR = sp.distance_matrix(Catalog,randomCatalog,2)
        distancesRR = sp.distance_matrix(randomCatalog,randomCatalog,2)
        # print(t.time()-ref)
        a,b = 20, 220
        dr = 10
        # r = np.array([0.5,0.7])
        step = (b-a)/10
        # zmax = 20
        r = np.array(np.linspace(a,b,int((b-a)/dr)+1))

        def numberPairs(Dist,r,dr):
            '''Dist is the distance matrix from two catalogs.
            r is the np.linspace with all the distances on which bins are centered
            dr is the width of a distance bin'''
            return(((Dist>r[:,None,None]-dr/2)&(Dist<r[:,None,None]+dr/2)).sum(axis=(1,2)))


        DD = numberPairs(distancesDD,r,dr)
        DR = numberPairs(distancesDR,r,dr)
        RR = numberPairs(distancesRR,r,dr)
        Xsi = (DD-2*DR+RR)/RR

        if save:
            XSIS.append(Xsi)
        # # plot
        if plot:
            plt.scatter(r,Xsi,marker ='+')
            #ref
            xref, yref = readtxt('xsi.txt')
            xref,yref = np.array(xref), np.array(yref)
            selection = (xref>=15)&(xref<=r[-1])
            xref,yref = xref[selection], yref[selection]
            plt.xlabel('Radial distance (Mpc)')
            plt.ylabel(r'$\xi(r)$')
            plt.plot(xref, yref,color = 'black')
            plt.legend(['Mine','Ref'])
            plt.show()
        np.savetxt('heavy files/binsCorrBigMCContinuous'+str(i)+'.txt',r)
        np.savetxt('heavy files/CorrBigMCContinuous'+str(i)+'.txt',Xsi)

    np.savetxt('heavy files/BigMCContinuousXSIs.txt',XSIS)

# #####################nbodykit#####################""
def convert_cartesian_to_sky_full_angle(X,Y,Z):
    RA = np.arctan2(X,Z)
    DEC = np.arcsin(np.nan_to_num(Y/np.sqrt(X**2 + Y**2 + Z**2)))
    r = np.sqrt(X**2 + Y**2 + Z**2)
    return(r,RA,DEC) #RA = phi, DEC = pi/2-theta


if mode == 'nbodykit':
    from nbodykit.algorithms.paircount_tpcf import tpcf
    from nbodykit.lab import ArrayCatalog
    import nbodykit.cosmology.cosmology as cosm

    print('Reading and compiling catalog...')

    Catalog = np.loadtxt('heavy files/BigCatalogContinuous15.txt')
    Catalog = np.nan_to_num(Catalog)
    # Catalog = np.loadtxt('thresholdcatalog.txt')
    n = Catalog.shape[0] #counts number
    print("Catalog size :",n)
    rmax = np.max(Catalog[:,0])
    data = ArrayCatalog({'RA': Catalog[:,2]*180/np.pi, 'DEC': Catalog[:,1]*180/np.pi, 'Redshift': Catalog[:,0], 'WEIGHT':np.ones(len(Catalog))})

    # # # # # #  random catalog creation
    print('Creating randomized catalog...')

    # ## randomCatalog from uniform
    rand_n = 3*n

    randomCatalog = random_Ball(radius=rmax,n=rand_n,mode = 'test') #ok functionnal
    # print('...and in spherical coordinates...')
    r, RA, DEC = convert_cartesian_to_sky_full_angle(randomCatalog[:,0],randomCatalog[:,1],randomCatalog[:,2])
    random_data = ArrayCatalog({'RA': RA*180/np.pi, 'DEC': DEC*180/np.pi,'Redshift' : r, 'WEIGHT':np.ones(len(r))})

    # randomCatalog = np.loadtxt('heavy files/RANDOMBigcatalogCorrelated.txt')
    # random_data = ArrayCatalog({'RA': randomCatalog[:,2]*180/np.pi, 'DEC': randomCatalog[:,1]*180/np.pi, 'Redshift': randomCatalog[:,0], 'WEIGHT':np.ones(len(randomCatalog))})

    print("Random catalog size :",randomCatalog.shape[0])

    print('Computing Landy and Szaslay estimator...')
    a,b = 15,225
    steps = 10
    bins = np.linspace(a,b,int((b-a)/steps)+1)

    class FakeCosmo(object):
                def comoving_distance(self, z):
                    return z
    C = FakeCosmo()

    Xsi = tpcf.SurveyData2PCF(mode='1d',data1=data,randoms1 = random_data, edges = bins,cosmo=C)

    print('Displaying results...')
    plt.subplot(121)
    res = Xsi.corr.data
    r = [el[1] for el in res]
    xsi = [el[0] for el in res]

    ## add errorbars thanks to MONTECARLO preliminary work:
    XsisMC = np.loadtxt('heavy files/XSIsBig.txt').T
    Cov = np.cov(XsisMC)
    # plt.imshow(np.corrcoef(XsisMC))
    # plt.colorbar()
    # plt.show()
    # plt.errorbar(r, xsi, yerr=np.sqrt(np.diag(Cov)),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
    ## http://www.python-simple.com/python-matplotlib/errorBars.php
    plt.scatter(r,xsi,color = 'blue',marker = '+',linewidths = 1.1)

    # np.savetxt('heavy files/binsCorrBig0.txt',r)
    # np.savetxt('heavy files/CorrBig0.txt',xsi)
    # np.savetxt('heavy files/stdCorrBig0.txt',np.sqrt(np.diag(Cov)))

    xref, yref = readtxt('xsi.txt')
    xref,yref = np.array(xref), np.array(yref)
    selection = (xref>=15)&(xref<=r[-1])
    xref,yref = xref[selection], yref[selection]
    # yref = 0.2**2*yref
    plt.xlabel('Radial distance (Mpc)')
    plt.ylabel(r'$\xi(r)$')
    plt.plot(xref, yref,color = 'black')
    plt.legend(['Mine','Ref'])

    try:
        plt.subplot(122)
        fc = interpolate.interp1d(r,xsi)
        xsi_2 = fc(xref)
        plt.plot(xref,np.log(1+xsi_2)/(np.log(1+yref)))
        plt.xlabel('Radial distance (Mpc)')
        plt.ylabel(r'$log(1+\xi_{mine}(r))/log(1+\xi_{ref}(r))$')
    except:
        print('oups')

    plt.show()

# ###################Monte Carlo##################
# mode = 'temp'
doubleCatalog = True

if mode == 'MC':
    from nbodykit.algorithms.paircount_tpcf import tpcf
    from nbodykit.lab import ArrayCatalog
    import nbodykit.cosmology.cosmology as cosm

    a,b = 15,225
    steps = 10
    bins = np.linspace(a,b,int((b-a)/steps)+1)

    class FakeCosmo(object):
                def comoving_distance(self, z):
                    return z
    C = FakeCosmo()
    XSIS = []

    for i in range(0,20):
        print('Iteration: ',i)
        # Catalog = np.loadtxt('heavy files/BigCatalogMC'+str(i)+'.txt')
        # Catalog = np.loadtxt('heavy files/BigCatalogContinuous'+str(i)+'.txt')
        Catalog = np.loadtxt('heavy files/BigDoubleCatalog'+str(i)+'.txt')

        n = Catalog.shape[0] #counts number
        rmax = np.max(Catalog[:,0])
        data = ArrayCatalog({'RA': Catalog[:,2]*180/np.pi, 'DEC': Catalog[:,1]*180/np.pi, 'Redshift': Catalog[:,0], 'WEIGHT':np.ones(len(Catalog))})

       
        if doubleCatalog:
            randomCatalog = np.loadtxt('heavy files/RANDOMBigDoubleCatalog'+str(i)+'.txt')
            random_data = ArrayCatalog({'RA': randomCatalog[:,2]*180/np.pi, 'DEC': randomCatalog[:,1]*180/np.pi, 'Redshift': randomCatalog[:,0], 'WEIGHT':np.ones(len(randomCatalog))})
        else:
            rand_n = 3*n
            randomCatalog = random_Ball(radius=rmax,n=rand_n,mode = 'test') #ok functionnal
            r, RA, DEC = convert_cartesian_to_sky_full_angle(randomCatalog[:,0],randomCatalog[:,1],randomCatalog[:,2])
            random_data = ArrayCatalog({'RA': RA*180/np.pi, 'DEC': DEC*180/np.pi,'Redshift' : r, 'WEIGHT':np.ones(len(r))})

        Xsi = tpcf.SurveyData2PCF(mode='1d',data1=data,randoms1 = random_data, edges = bins,cosmo=C)

        res = Xsi.corr.data
        r = [el[1] for el in res]
        xsi = [el[0] for el in res]
        # print(xsi)
        XSIS.append(xsi)

        # np.savetxt('heavy files/binsCorrBigMC'+str(i)+'.txt',r)
        # np.savetxt('heavy files/CorrBigMC'+str(i)+'.txt',xsi)

        # np.savetxt('heavy files/binsCorrBigMCContinuous'+str(i)+'.txt',r)
        # np.savetxt('heavy files/CorrBigMCContinuous'+str(i)+'.txt',xsi)

        np.savetxt('heavy files/binsCorrBigDouble'+str(i)+'.txt',r)
        np.savetxt('heavy files/CorrBigDouble'+str(i)+'.txt',xsi)

        plt.scatter(r,xsi,marker = '+')
        plt.plot(r,integralXsi(r,univ = cosmo.Cosmology()))
        plt.show()
    # np.savetxt('heavy files/BigMCCOntinuousXSIs.txt',XSIS)
    np.savetxt('heavy files/BigDoubleCatalogXSIs.txt',XSIS)


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
# XSIS = []
# # XsisMC = np.loadtxt('heavy files/BigMCXSIs.txt').T
# # XsisMC = np.loadtxt('heavy files/BigMCContinuousXSIs.txt').T
# # Cov = np.cov(XsisMC)
# # np.savetxt('heavy files/stdCorrBigMCContinuous.txt',np.sqrt(np.diag(Cov)))

# for i in range(20):
#     try:
#         # r = np.loadtxt('heavy files/binsCorrBigMC'+str(i)+'.txt')
#         # xsi = np.loadtxt('heavy files/CorrBigMC'+str(i)+'.txt')
#         # r = np.loadtxt('heavy files/binsCorrBigMCContinuous'+str(i)+'.txt')
#         # xsi = np.loadtxt('heavy files/CorrBigMCContinuous'+str(i)+'.txt')
#         r = np.loadtxt('heavy files/binsCorrBigMCContinuous'+str(i)+'.txt')
#         xsi = np.loadtxt('heavy files/CorrBigCorrelation'+str(i)+'.txt')
#         XSIS.append(xsi)
#         # # std = np.loadtxt('heavy files/stdCorrBigMC.txt')
#         # std = np.loadtxt('heavy files/stdCorrBigMCContinuous.txt')
#         # # xsi = xsi/(0.2**2)
#         # plt.errorbar(r, xsi, yerr=std,fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
#         plt.scatter(r,xsi,color = 'blue',marker = '+',linewidths = 1.1)

#         xref, yref = readtxt('xsi.txt')
#         xref,yref = np.array(xref), np.array(yref)
#         selection = (xref>=15)&(xref<=r[-1])
#         xref,yref = xref[selection], yref[selection]
#         # yref = 0.2**2*yref
#         plt.xlabel('Radial distance (Mpc)')
#         plt.ylabel(r'$\xi(r)$')
#         plt.plot(xref, yref,color = 'black')
#         plt.legend(['Mine','Ref'])
#         plt.show()
#     except:
#         print('no such file')
#         continue


# # plt.show()
# np.savetxt('heavy files/BigCorrelationXSIs.txt',XSIS)


# ####simple plot of mean (mean)
# XSIS = []
# # XsisMC = np.loadtxt('heavy files/BigMCXSIs.txt')
# # XsisMC = np.loadtxt('heavy files/BigMCContinuousXSIs.txt')
# XsisMC = np.loadtxt('heavy files/BigCorrelationXSIs.txt')
# XSIS = np.array(XsisMC).T
# meanXSIS = np.mean(XSIS,axis = 1)
# print(np.cov(XSIS).shape)
# Cov = np.cov(XSIS)/XSIS.shape[0]

# # np.savetxt('heavy files/CorrBigMCmean.txt',meanXSIS)
# # np.savetxt('heavy files/CorrBigMCContinuousmean.txt',meanXSIS)
# np.savetxt('heavy files/CorrBigCorrelationmean.txt',meanXSIS)
# #ref
# xr = np.linspace(15,220,50)
# y = integralXsi(xr,cosmo.Cosmology())
# plt.plot(xr, y,color = 'black',linestyle = '--')

# r = np.loadtxt('heavy files/binsCorrBigMC0.txt')
# plt.errorbar(r,meanXSIS, yerr= np.sqrt(Cov.diagonal()),ecolor= 'red', fmt = 'none',capsize = 3,elinewidth = 0.7,capthick=0.7) # Be careful those are not error bars but only the delta delta values dispersions
# plt.scatter(r,meanXSIS,linewidths=1.1,color = 'blue',marker = '+')

# plt.legend(['Theoretical','Box (bins)'])
# plt.tight_layout()
# plt.show()



