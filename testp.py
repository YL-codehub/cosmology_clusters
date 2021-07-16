import numpy as np
import matplotlib.pyplot as plt
import csv

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

def transfer_Function_BKKS(k,Om = 0.3,h= 0.7):
    '''Transfer function at wavenumber k (units Mpc^-1)'''
    theta = 1
    q = k * theta ** 0.5 / (Om * h ** 2)  # Mpc
    res = (np.log(1 + 2.34 * q) / (2.34 * q)) * np.power(1 + 3.89 * q + np.power(16.1 * q, 2) + np.power(5.46 * q, 3) + np.power(6.71 * q, 4), -0.25)
    res = np.nan_to_num(res,nan = 1)
    return res


def window(y):
    '''Window function in Fourier space, the product with which allows to get rid of low values of radius or mass'''
    return (3 * (np.sin(y) / y - np.cos(y)) / np.power(y, 2))

import scipy.integrate as intg
# As = sigma8 ** 2 * (2 * np.pi) ** 3 / (4*np.pi*intg.quad(lambda K : K **  ns *  transfer_Function_BKKS(K) ** 2 * abs( window(K * 8/ h))** 2 * K ** 2, 0, np.inf,limit = 100)[0])
# print(As)
As = 6027309.4271296235

def initial_Power_Spectrum_BKKS(K):
    '''spatial part of Power spectrum (units = Mpc^3) at wavenumber k (units = Mpc^-1)'''
    return (As * np.multiply(np.power(K,ns),np.power(transfer_Function_BKKS(K),2)))

#==================================
def main():
#==================================


    nc = 128       # define how many cells your box has
    boxlen = nc*20       # define length of box (Mpc)
    dx = boxlen/nc          # get size of a cell (Mpc), 20Mpc gives ~ 8h^-1 Mpc sphere

    # get overdensity field
    delta = np.random.normal(0, 1, size=(nc, nc, nc))
    delta_k = np.fft.rfftn(delta)
    # delta_k = delta_k/np.sqrt(np.mean(np.abs(delta_k))**2)
    # print(np.mean(np.abs(delta_k))**2)
    # get 3d array of index integer distances to k = (0, 0, 0), ie compute k values' grid in Fourier space
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist #²
    dist_z *= dist_z #²
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

#ajout:
    dk = 2*np.pi/boxlen
    kMpc = dist_3d*dk #Mpc^-1

    # Compute spectrum
    P_BKKS = initial_Power_Spectrum_BKKS(kMpc)
    sqP_BKKS = np.sqrt(initial_Power_Spectrum_BKKS(kMpc))
    # print(P_BKKS)
    Spectrum = np.multiply(delta_k,sqP_BKKS)

    # print(Spectrum)

    # Back to real space
    delta_new = np.fft.irfftn(Spectrum)

    # plot overdensity constrast
    fig, axs = plt.subplots(2, 2)
    axs[0,0].imshow(delta[nc//2,:,:])
    axs[0,0].set_title('Initial white noise')


    axs[1, 0].imshow(delta_new[nc // 2, :, :])
    axs[1, 0].set_title(r'Overensity contrast at z = 0, $\sigma_8 =$ '+str(np.std(delta_new).round(2)))

    # Check Spectrum (and so the units !)

    xref, yref = readtxt('pk_bbks.txt')
    axs[0,1].plot(xref, yref, color='red')

    distances, _ = np.unique(kMpc, return_inverse=True)
    Pk = np.bincount(_, weights=np.abs(P_BKKS.ravel())) / np.bincount(_)
    axs[0, 1].scatter(distances, Pk, marker='+',color = 'blue')
    nPk = np.bincount(_, weights=np.abs(np.multiply(P_BKKS,abs(delta_k)**2).ravel()))/np.bincount(_)
    axs[0,1].scatter(distances, nPk, marker = '+',color = 'green')

    axs[0,1].legend(['Reference', 'Non noisy mine', 'Noisy Mine'])
    axs[0,1].semilogy()
    axs[0,1].semilogx()
    axs[0,1].set_xlim([1e-5,1e2])
    axs[0,1].grid()
    axs[0,1].set_title('Induced Spectrum')
    axs[0,1].set_xlabel('Wave number '+r'$k$ $(Mpc^{-1}$)')
    axs[0,1].set_ylabel(r'$P(k)$')
    plt.show()

#     # Computing correlation. # NE PAS TOUT CALCULER
#     val = {}
#     n =1
#     for i1 in range(nc):
#         for i2 in range(nc):
#             for j1 in range(nc):
#                 for j2 in range(nc):
#                     for k1 in range(nc):
#                         for k2 in range(nc):
#                             ind = int((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
#                             try:
#                                 val[ind].append(delta_new[i1, j1, k1] * delta_new[i2, j2, k2])
#                             except KeyError:
#                                 val[ind] = [delta_new[i1, j1, k1] * delta_new[i2, j2, k2]]

#     #     # Computing correlation. # NE PAS TOUT CALCULER
# #     val = {}
# #     n =1
#       points = np.random.randint(0,nc,size = (2000,6))
# #     for pair in points:
#           i1,i2,j1,j2,k1,k2 = pair
# #         ind = int((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
# #         try:
# #             val[ind].append(delta_new[i1, j1, k1] * delta_new[i2, j2, k2])
# #         except KeyError:
# #             val[ind] = [delta_new[i1, j1, k1] * delta_new[i2, j2, k2]]

#     X = []
#     Xsi = []
#     for key in val.keys():
#         if len(val[key]) > nc**2/2:
#             Xsi.append(np.mean(val[key])) #biased correlation estimation,
#             X.append(np.sqrt(key) * dx)
#         # Xsi.append(np.sum(val[key])/(len(val[key])-np.sqrt(key))) #unbiased correlation estimation, estimateur est rarement utilisé car sa variance est très élevée pour les valeurs de k proches de N, et en général moins bon que le cas biaisé
#     # plt.scatter(X,Xsi,linewidths=0.05,color = 'red')
#     plt.ylabel('Correlation function estimation')
#     plt.xlabel('Radial distance (Mpc)')
#
#
#     # Correlation bins
#     nbins = 50
#     beginbins = np.linspace(0,np.max(X),nbins)
#     bins = []
#     stdbins = []
#     for i in range(nbins-1):
#         tempbin = []
#         for j in range(len(X)):
#             if X[j]>= beginbins[i] and X[j]< beginbins[i+1]:
#                 tempbin.append(Xsi[j])
#         bins.append(np.mean(tempbin))
#         stdbins.append(np.std(tempbin))
#     # plt.scatter(beginbins[:-1]+0.5*np.max(X)/nbins,bins,color = 'blue',marker = '+')
#     plt.errorbar(beginbins[:-1]+0.5*np.max(X)/nbins,bins, yerr=stdbins,ecolor= 'red')
#
# ## Reference correlation
#     xref, yref = readtxt('xsi.txt')
#     plt.scatter(xref, yref,linewidths=0.05,color = 'green')
#     plt.legend(['Mine','Ref'])
#
#     plt.show()










#==================================
if __name__ == "__main__":
#==================================

    main()

# #non real = not a half grid
# dist = np.minimum(np.arange(nc), np.arange(nc, 0, -1))
# # dist_z = np.arange(nc // 2 + 1)
# dist *= dist  # ²
# # dist_z *= dist_z  # ²
# dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist)
# print(dist)

# get unique distances and index which any distance stored in dist_3d
# will have in "distances" array ie supprime doublons, distance = listes des valeurs uniques, _ est dist_3d mais avec les indices correspondants aux valeurs dans distances
# distances, _ = np.unique(dist_3d, return_inverse=True)
# distances, _ = np.unique(kMpc, return_inverse=True)

# average P(kx, ky, kz) to P(|k|) ie
## np.bincount(_) est compte le nombre d'occurences de chaque distance (dans distances)
## np.bincount(_, weights=Pk_field.ravel())/np.bincount(_) donne la moyenne de Pk pour chaque distance
# Pk = np.bincount(_, weights=Pk_field.ravel())/np.bincount(_)
# Pk = np.bincount(_, weights=np.abs(P_BKKS.ravel()))/np.bincount(_)
# Pk = np.bincount(_, weights=np.abs(P_BKKS.ravel()))/np.bincount(_)

# # compute "phyical" values of k
# dk = 2*np.pi/boxlen
# k = distances*dk #Mpc^-1

# plot results
# fig = plt.figure(figsize=(9,6))
# ax1 = fig.add_subplot(111)
# ax1.scatter(distances, Pk, label=r'$P(\mathbf{k})$',marker = '+')
# plt.semilogx()
# plt.semilogy()