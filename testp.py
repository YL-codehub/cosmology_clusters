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

# As = sigma8 ** 2 * (2 * m.pi) ** 3 / (4*m.pi*intg.quad(lambda K : K **  ns *  transfer_Function_BKKS(K) ** 2 * abs( window(K * 8/ h))** 2 * K ** 2, 0, m.inf,limit = 100)[0])
As = 6027309.4271296235

def initial_Power_Spectrum_BKKS(K):
    '''spatial part of Power spectrum (units = Mpc^3) at wavenumber k (units = Mpc^-1)'''
    return (As * np.multiply(np.power(K,ns),np.power(transfer_Function_BKKS(K),2)))

#==================================
def main():
#==================================


    nc = 20              # define how many cells your box has
    boxlen = 1*20           # define length of box (Mpc)
    # Lambda = boxlen/4.0     # define an arbitrary wave length of a plane wave
    dx = boxlen/nc          # get size of a cell (Mpc), 20Mpc gives ~ 8h^-1 Mpc sphere

    # create plane wave density field
    # density_field = np.zeros((nc, nc, nc), dtype='float')
    # for x in range(density_field.shape[0]):
    #     density_field[x,:,:] = np.cos(2*np.pi*x*dx/Lambda)

    # get overdensity field
    # delta = density_field/np.mean(density_field) - 1
    delta = np.random.normal(0, 1, size=(nc, nc, nc))
# get P(k) field: explot fft of data that is only real, not complex
#     delta_k = np.abs(np.fft.rfftn(delta).round())
    delta_k = np.fft.rfftn(delta)


    #test rfftn
    # delta_k = np.fft.rfftn(delta)
    # delta_k_r = np.fft.irfftn(delta_k)
    # plt.imshow(delta[nc//2,:,:]-delta_k_r[nc//2,:,:])
    # plt.colorbar()

    # print(delta_k.shape)
    # Pk_field =  delta_k**2

    # get 3d array of index integer distances to k = (0, 0, 0), ie compute k values' grid in Fourier space
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist #²
    dist_z *= dist_z #²
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

    # #non real = not a half grid
    # dist = np.minimum(np.arange(nc), np.arange(nc, 0, -1))
    # # dist_z = np.arange(nc // 2 + 1)
    # dist *= dist  # ²
    # # dist_z *= dist_z  # ²
    # dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist)
    # print(dist)

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

    # print(p_bkks)
    # print(delta_new[nc//2+1,:,:])
    # plt.imshow(delta_new[nc//2,:,:])

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

    # # plot expected peak:
    # # k_peak = 2*pi/lambda, where we chose lambda for our planar wave earlier
    # ax1.plot([2*np.pi/Lambda]*2, [Pk.min()-1, Pk.max()+1], label='expected peak')
    # ax1.legend()

    ## Computing correlation.
    val = {}
    n =1
    for i1 in range(nc):
        for i2 in range(nc):
            for j1 in range(nc):
                for j2 in range(nc):
                    for k1 in range(nc):
                        for k2 in range(nc):
                            ind = int((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
                            try:
                                val[ind].append(delta_new[i1, j1, k1] * delta_new[i2, j2, k2])
                            except KeyError:
                                val[ind] = [delta_new[i1, j1, k1] * delta_new[i2, j2, k2]]
    X = []
    Xsi = []
    for key in val.keys():
        if len(val[key]) > nc**2/2:
            Xsi.append(np.mean(val[key])) #biased correlation estimation,
            X.append(np.sqrt(key) * dx)
        # Xsi.append(np.sum(val[key])/(len(val[key])-np.sqrt(key))) #unbiased correlation estimation, estimateur est rarement utilisé car sa variance est très élevée pour les valeurs de k proches de N, et en général moins bon que le cas biaisé
    # plt.scatter(X,Xsi,linewidths=0.05,color = 'red')
    plt.ylabel('Correlation function estimation')
    plt.xlabel('Radial distance (Mpc)')


    # Correlation bins
    nbins = 50
    beginbins = np.linspace(0,np.max(X),nbins)
    bins = []
    stdbins = []
    for i in range(nbins-1):
        tempbin = []
        for j in range(len(X)):
            if X[j]>= beginbins[i] and X[j]< beginbins[i+1]:
                tempbin.append(Xsi[j])
        bins.append(np.mean(tempbin))
        stdbins.append(np.std(tempbin))
    # plt.scatter(beginbins[:-1]+0.5*np.max(X)/nbins,bins,color = 'blue',marker = '+')
    plt.errorbar(beginbins[:-1]+0.5*np.max(X)/nbins,bins, yerr=stdbins,ecolor= 'red')

### Reference correlation
    # xref, yref = readtxt('xsi.txt')
    # plt.scatter(xref, yref,linewidths=0.05,color = 'green')
    # plt.legend(['Mine','Ref'])

    plt.show()










#==================================
if __name__ == "__main__":
#==================================

    main()
