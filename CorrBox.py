### Fast and rigorous method to get correlation in the box, also implemented in gen.py

import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.spatial.kdtree import distance_matrix
import randPoints as rP
import scipy.stats as st
import Cosmological_tools as cosmo


nc = 256  # define how many cells your box has
boxlen = nc*20       # define length of box (Mpc)
dx = boxlen/nc          # get size of a cell (Mpc), 20Mpc gives ~ 8h^-1 Mpc sphere

def window(y,mode = 'np'):
        '''Window function in Fourier space, the product with which allows to get rid of low values of radius or mass'''
        if mode == 'default':
            try:
                if y <1e-7:
                    return(1)
                elif y == m.inf:
                    return(0)
                else:
                    return(3*(m.sin(y)/y-m.cos(y))/y**2)
            except ValueError:
                print(y)
        elif mode == 'np':
            return (3 * (np.sin(y) / y - np.cos(y)) / np.power(y, 2))

def integralXsi(R,univ,a = 1e-7, b= 1e3,n = 100000):
    K = np.array([np.linspace(a, b, n)])
    dK = (b - a) / n
    X = np.array([R]).T
    F = univ.initial_Power_Spectrum_BKKS(K, mode='np') * np.sin(K * X) / (K * X) * K ** 2 / (2 * np.pi ** 2)
    xsi_model = np.sum(F, axis=1) * dK
    return(xsi_model)

def integralXsiMode(R,univ,a = 1e-7, b= 1e3,n = 100000,mode = 0):
    K = np.array([np.linspace(a, b, n)])
    dK = (b - a) / n
    X = np.array([R]).T
    prod = np.kron(K, X)
    if mode == 0:
        F = univ.initial_Power_Spectrum_BKKS(K, mode='np') * np.sin(K * X) / (K * X) * K ** 2 / (2 * np.pi ** 2)
    else: #mode = R smoother
        R = mode
        F = np.multiply(univ.initial_Power_Spectrum_BKKS(K, mode='np'),np.power(window(K * R, mode = 'np'),2)) * np.sin(K * X) / (K * X) * K ** 2 / (2 * np.pi ** 2)
    xsi_model = np.sum(F, axis=1) * dK
    return(xsi_model)

a,b = 20,220 #Mpc
BinStep = 0.5 #pixels (x20 Mpc)


############# Method 1 : iterative ##############
# delta_new = np.fromfile('heavy files/box2nc'+str(nc)+'dx'+str(int(dx)))
# delta_new = np.reshape(delta_new,(nc,nc,nc))

# val = {}
# N = 1000
# # draw in a sub-box
# points = np.random.uniform(nc//2-b/(2*dx),nc//2+b/(2*dx),size = (N,3))

# for i in range(len(points)):
#     for j in range(i,len(points)):
#         i1, j1, k1 = points[i,:]
#         i2, j2, k2 = points[j,:]
#         norm = np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
#         q = np.round(norm/BinStep)
#         ind = q*BinStep
#         try:
#             val[ind].append(delta_new[int(i1), int(j1), int(k1)] * delta_new[int(i2), int(j2), int(k2)])
#         except KeyError:
#             val[ind] = [delta_new[int(i1), int(j1), int(k1)] * delta_new[int(i2), int(j2), int(k2)]]

# #Binning vfinale :
# print('...Binning...')
# bins = []
# Xsi = []
# stdbins = []
# for key in val.keys():
#     if len(val[key]) > 0:#nc**2/2:
#         if key*dx<=b+BinStep/2 and key*dx>=a-BinStep/2 : #do not count over 215 Mpc and under 10Mpc (non-linear regime)
#             Xsi.append(np.mean(val[key])) #biased correlation estimation,
#             bins.append(key*dx)
#             stdbins.append(np.std(val[key]))

################# Method 2 : fast ###############
import scipy.spatial as sp


def XsiEval(r, Dist,xsis,dr):
    '''Dist is the distance matrix from two catalogs.
    xsis = kron
    r is the np.linspace with all the distances on which bins are centered
    dr is the width of a distance bin'''
    # eligible = np.multiply((Dist>r[:,None,None]-dr/2)&(Dist<r[:,None,None]+dr/2),xsis)
    eligible = np.tril(np.multiply((Dist>r-dr/2)&(Dist<=r+dr/2),xsis))
    eligible = eligible[np.nonzero(eligible)] 
    return (np.mean(eligible), np.std(eligible))

# Monte Carlo or not if many

XSIS = []
dr = BinStep*dx #Mpc
bins = np.array(np.linspace(a,b,int((b-a)/dr)+1))
N = 4000
p = 20 # number of boxes
for i in range(0):
    print('Iteration : ',i)
    delta_new = np.fromfile('heavy files/box'+str(i)+'nc'+str(nc)+'dx'+str(int(dx)))
    delta_new = np.reshape(delta_new,(nc,nc,nc))

    # points = np.random.uniform(nc//2-b/(2*dx),nc//2+b/(2*dx),size = (N,3)) #pixels
    ### I take a sub-cube with 10 times more 
    n = np.power(10*N,1/3)
    points = np.random.uniform(nc//2-n/2,nc//2+n/2,size = (N,3)) #pixels

    distancesMatrix = sp.distance_matrix(points,points,2)*dx #Mpc
    Delta = np.array([delta_new[points[:,0].astype(int),points[:,1].astype(int),points[:,2].astype(int)]]).T
    xsis = np.kron(Delta,Delta.T)

    Xsi = []
    stdbins = []

    for r in bins:
        xsi, std = XsiEval(r,Dist = distancesMatrix,xsis = xsis,dr = dr)
        Xsi.append(xsi)
        # stdbins.append(std)
    XSIS.append(Xsi)

# np.savetxt('heavy files/BoxXSIS.txt',XSIS)

XSIS = np.loadtxt('heavy files/BoxXSIS.txt')
XSIS = np.array(XSIS).T
Cov = np.cov(XSIS)/p
meanXSIS = np.mean(XSIS,axis = 1)

# np.savetxt('heavy files/BoxbinsXSIS.txt', bins)
# np.savetxt('heavy files/BoxmeanXSIS.txt',meanXSIS)
# np.savetxt('heavy files/BoxstdXSIS.txt',np.sqrt(Cov.diagonal()))

#ref
xr = np.linspace(15,220,50)
y = integralXsi(xr,cosmo.Cosmology())
plt.plot(xr, y,color = 'black',linestyle = '--')

plt.errorbar(bins,meanXSIS, yerr= np.sqrt(Cov.diagonal()),ecolor= 'red', fmt = 'none',capsize = 3,elinewidth = 0.7,capthick=0.7) # Be careful those are not error bars but only the delta delta values dispersions
plt.scatter(bins,meanXSIS,linewidths=1.1,color = 'blue',marker = '+')

plt.legend(['Theoretical','Box (bins)'])
plt.tight_layout()
plt.show()

#################### Plot ########################
                
# plt.scatter(bins,Xsi,linewidths=0.5,color = 'blue',marker = '+', alpha=0.6)
# # # # # plt.errorbar(bins,Xsi, yerr=stdbins,ecolor= 'red') # Be careful those are not error bars but only the delta delta values dispersions


# xr = np.linspace(15,220,50)
# y = integralXsi(xr,cosmo.Cosmology())
# plt.plot(xr, y,color = 'red',linestyle = '--')
# plt.legend(['Mine (bins)','Ref'])
# plt.tight_layout()
# plt.show()

################ refine convoluted model ##########""

def objective_function(R):
    r = np.array(np.loadtxt('heavy files/BoxbinsXSIS.txt'))
    xsi_data = np.array(np.loadtxt('heavy files/BoxmeanXSIS.txt'))
    std = np.array(np.loadtxt('heavy files/BoxstdXSIS.txt'))
    xsi_convolutedR = integralXsiMode(r,cosmo.Cosmology(),mode = R)
    return np.sum(np.power((xsi_convolutedR-xsi_data)/std,2))

# import scipy.optimize as opt
# res = opt.minimize(objective_function, [0],options={ 'disp': True}, bounds = ((0,15),)).x
# print('R = ',res)

# # plot
# r = np.linspace(15,230,50)
# xsi_th = integralXsiMode(r,cosmo.Cosmology())
# xsi_convolutedR= integralXsiMode(r,cosmo.Cosmology(),mode = res)
# plt.plot(r,xsi_th, color = 'grey',linewidth = 1)
# plt.plot(r,xsi_convolutedR,color = 'black',linestyle = '--',linewidth = 1)
# plt.title('Theoretical space correlation functions')
# plt.xlabel('Radial distance (Mpc)')
# plt.ylabel('Correlation function '+r'$\xi$')

# #effective in the box :
# x = np.array(np.loadtxt('heavy files/BoxbinsXSIS.txt'))
# y = np.array(np.loadtxt('heavy files/BoxmeanXSIS.txt'))
# std = np.array(np.loadtxt('heavy files/BoxstdXSIS.txt'))
# plt.errorbar(x,y, yerr=std,fmt='none',ecolor= 'red',alpha=0.6)
# plt.scatter(x,y, color = 'blue',marker = '+')
# plt.legend([r'$\xi$',str(np.round(res,2)[0])+'Mpc-top-hat convoluted '+r'$\xi$','Box'])
       
# plt.show()