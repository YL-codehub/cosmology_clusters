import numpy as np
import matplotlib.pyplot as plt
import csv
<<<<<<< HEAD
=======
import randPoints as rP
import scipy.stats as st
import scipy.spatial as sp
import Cosmological_tools as cosmo
>>>>>>> temp

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
<<<<<<< HEAD
    with open(file, newline='\n') as csvfile1:
=======
    with open('Vérifications fonctions/'+file, newline='\n') as csvfile1:
>>>>>>> temp
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
<<<<<<< HEAD
=======

>>>>>>> temp

    # Array integral mode :
def integralXsi(R,univ,a = 1e-7, b= 1e3,n = 100000):
    K = np.array([np.linspace(a, b, n)])
    dK = (b - a) / n
    X = np.array([R]).T
    F = univ.initial_Power_Spectrum_BKKS(K, mode='np') * np.sin(K * X) / (K * X) * K ** 2 / (2 * np.pi ** 2)
    xsi_model = np.sum(F, axis=1) * dK
    return(xsi_model)

def window(y):
    '''Window function in Fourier space, the product with which allows to get rid of low values of radius or mass'''
    return (3 * (np.sin(y) / y - np.cos(y)) / np.power(y, 2))

<<<<<<< HEAD
import scipy.integrate as intg
# As = sigma8 ** 2 * (2 * np.pi) ** 3 / (4*np.pi*intg.quad(lambda K : K **  ns *  transfer_Function_BKKS(K) ** 2 * abs( window(K * 8/ h))** 2 * K ** 2, 0, np.inf,limit = 100)[0])
=======
def XsiEval(r, Dist,xsis,dr):
    '''Dist is the distance matrix from two catalogs.
    xsis = kron
    r is the np.linspace with all the distances on which bins are centered
    dr is the width of a distance bin'''
    # eligible = np.multiply((Dist>r[:,None,None]-dr/2)&(Dist<r[:,None,None]+dr/2),xsis)
    eligible = np.tril(np.multiply((Dist>r-dr/2)&(Dist<=r+dr/2),xsis))
    eligible = eligible[np.nonzero(eligible)] 
    return (np.mean(eligible), np.std(eligible))

import scipy.integrate as intg
# As = sigma8 ** 2 * (2 * np.pi) ** 3 / (4*np.pi*intg.quad(lambda K : K **  ns *  transfer_Function_BKKS(K) ** 2 * abs( window(K * 8/ h))** 2 * K ** 2, 0, np.inf,limit = 500)[0])
>>>>>>> temp
# print(As)
As = 6027309.4271296235

def initial_Power_Spectrum_BKKS(K):
    '''spatial part of Power spectrum (units = Mpc^3) at wavenumber k (units = Mpc^-1)'''
    return (As * np.multiply(np.power(K,ns),np.power(transfer_Function_BKKS(K),2)))

#==================================
<<<<<<< HEAD
def main():
#==================================


    nc = 20      # define how many cells your box has
=======
def main(saveCorr = -1):
#==================================


    nc = 256  # define how many cells your box has
>>>>>>> temp
    boxlen = nc*20       # define length of box (Mpc)
    dx = boxlen/nc          # get size of a cell (Mpc), 20Mpc gives ~ 8h^-1 Mpc sphere

    # get overdensity field
    delta = np.random.normal(0, 1, size=(nc, nc, nc))
<<<<<<< HEAD
    # delta_k = np.fft.rfftn(delta)
    delta_k = np.fft.rfftn(delta)/(nc**(3/2)) #equivalent to norm = "ortho")
    # delta_k = np.random.normal(0, 1, size=(nc, nc, nc//2+1))
    # delta_k = delta_k/np.sqrt(np.mean(np.abs(delta_k))**2)
    # print(np.mean(np.abs(delta_k))**2)
    # get 3d array of index integer distances to k = (0, 0, 0), ie compute k values' grid in Fourier space
=======
    print('ffting...')
    delta_k = np.fft.rfftn(delta)/(nc**(3/2)) #equivalent to norm = "ortho")

    # get 3d array of index integer distances to k = (0, 0, 0), ie compute k values' grid in Fourier space
    print('Spectrum incoming...')
>>>>>>> temp
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist #²
    dist_z *= dist_z #²
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

<<<<<<< HEAD
#ajout:
=======
>>>>>>> temp
    dk = 2*np.pi/boxlen
    kMpc = dist_3d*dk #Mpc^-1

    # Compute spectrum
    P_BKKS = initial_Power_Spectrum_BKKS(kMpc)
    sqP_BKKS = np.sqrt(initial_Power_Spectrum_BKKS(kMpc))
    # print(P_BKKS)
    Spectrum = np.multiply(delta_k,sqP_BKKS)

    # print(Spectrum)

    # Back to real space
<<<<<<< HEAD
    delta_new = np.fft.irfftn(Spectrum, norm = "ortho")*(1/dx)**(3/2) # converting sqrt(P(k))**3 (Mpc^3/2) to cells^3/2
    # delta_new = np.fft.irfftn(Spectrum)/(2*np.pi)**3

    # plot overdensity constrast
=======
    print('inverse ffting...')
    delta_new = np.fft.irfftn(Spectrum, norm = "ortho")*(1/dx)**(3/2) # converting sqrt(P(k))**3 (Mpc^3/2) to cells^3/2
    
    ## Save calculus
    delta_new.tofile('heavy files/box2nc'+str(int(nc))+'dx'+str(int(dx)))
    # delta_new2 = np.fromfile('box nc='+str(nc)+' and dx = '+str(dx))
    # delta_new2 = np.reshape(delta_new2,(nc,nc,nc))

    # plot overdensity constrast
    print('Plotting contrasts slices...')
>>>>>>> temp
    fig, axs = plt.subplots(2, 2)
    axs[0,0].imshow(delta[nc//2,:,:])
    axs[0,0].set_title('Initial white noise')


    axs[1, 0].imshow(delta_new[nc // 2, :, :])
<<<<<<< HEAD
    axs[1, 0].set_title(r'Overensity contrast at z = 0, $\sigma_8 =$ '+str(np.std(delta_new).round(2)))

    # Check Spectrum (and so the units !)

=======
    axs[1, 0].set_title(r'$\delta$'+' map, '+ r'$\sigma =$'+str(np.std(delta_new).round(2)))

    # Check Spectrum (and so the units !)
    print('Plotting Spectrum...')
>>>>>>> temp
    xref, yref = readtxt('pk_bbks.txt')
    axs[0,1].plot(xref, yref, color='red')

    distances, _ = np.unique(kMpc, return_inverse=True)
    Pk = np.bincount(_, weights=np.abs(P_BKKS.ravel())) / np.bincount(_)
<<<<<<< HEAD
    axs[0, 1].scatter(distances, Pk, marker='+',color = 'blue')
    nPk = np.bincount(_, weights=np.abs(np.multiply(P_BKKS,abs(delta_k)**2).ravel()))/np.bincount(_)
    axs[0,1].scatter(distances, nPk, marker = '+',color = 'green')

    axs[0,1].legend(['Reference', 'Non noisy mine', 'Noisy Mine'])
=======
    # axs[0, 1].scatter(distances, Pk, marker='+',color = 'blue')
    nPk = np.bincount(_, weights=np.abs(np.multiply(P_BKKS,abs(delta_k)**2).ravel()))/np.bincount(_)
    axs[0,1].scatter(distances, nPk, marker = '+',color = 'green')

    # axs[0,1].legend(['Reference', 'Non noisy mine', 'Noisy Mine'])
    axs[0,1].legend(['Theoretical', 'Box (noisy)'])
>>>>>>> temp
    axs[0,1].semilogy()
    axs[0,1].semilogx()
    axs[0,1].set_xlim([1e-5,1e2])
    axs[0,1].grid()
<<<<<<< HEAD
    axs[0,1].set_title('Induced Spectrum')
=======
    axs[0,1].set_title('Spectrum space')
>>>>>>> temp
    axs[0,1].set_xlabel('Wave number '+r'$k$ $(Mpc^{-1}$)')
    axs[0,1].set_ylabel(r'$P(k)$')
    # plt.show()

<<<<<<< HEAD
    # Computing correlation. # NE PAS TOUT CALCULER
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

    #     # Computing correlation. # NE PAS TOUT CALCULER
#     val = {}
#     n =1
    #   points = np.random.randint(0,nc,size = (2000,6))
#     for pair in points:
        #   i1,i2,j1,j2,k1,k2 = pair
#         ind = int((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
#         try:
#             val[ind].append(delta_new[i1, j1, k1] * delta_new[i2, j2, k2])
#         except KeyError:
#             val[ind] = [delta_new[i1, j1, k1] * delta_new[i2, j2, k2]]

    X = []
    Xsi = []
    for key in val.keys():
        if len(val[key]) > nc**2/2:
            Xsi.append(np.mean(val[key])) #biased correlation estimation,
            X.append(np.sqrt(key) * dx)
        # Xsi.append(np.sum(val[key])/(len(val[key])-np.sqrt(key))) #unbiased correlation estimation, estimateur est rarement utilisé car sa variance est très élevée pour les valeurs de k proches de N, et en général moins bon que le cas biaisé
    # plt.scatter(X,Xsi,linewidths=0.05,color = 'red')
    axs[1,1].set_ylabel('Correlation function estimation')
    axs[1,1].set_xlabel('Radial distance (Mpc)')


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
    axs[1,1].errorbar(beginbins[:-1]+0.5*np.max(X)/nbins,bins, yerr=stdbins,ecolor= 'red')

## Reference correlation
    xref, yref = readtxt('xsi.txt')
    axs[1,1].scatter(xref, yref,linewidths=0.05,color = 'green')
    axs[1,1].legend(['Ref','Mine'])

    plt.show()










#==================================
if __name__ == "__main__":
#==================================

    main()
=======
    print('Computing correlation...')
    print('...Evaluating...')
    # # # Computing correlation. # NE PAS TOUT CALCULER, just a sub-box
    # val = {}
    # n =1
    # p = 10
    # mid = nc//2
    # for i1 in range(mid-p,mid+p):
    #     for i2 in range(mid-p,mid+p):
    #         for j1 in range(mid-p,mid+p):
    #             for j2 in range(mid-p,mid+p):
    #                 for k1 in range(mid-p,mid+p):
    #                     for k2 in range(mid-p,mid+p):
    #                         ind = int((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
    #                         try:
    #                             val[ind].append(delta_new[i1, j1, k1] * delta_new[i2, j2, k2])
    #                         except KeyError:
    #                             val[ind] = [delta_new[i1, j1, k1] * delta_new[i2, j2, k2]]

    #     # Computing correlation. # NE PAS TOUT CALCULER
    
    # val = {}
    # points = np.random.randint(0,nc/2,size = (10000000,6))
    # # normsup = 15#nc//2
    # # points =rP.randomPoints(1, normsup ,100,n = 10000)
    # # points = points+nc//2-np.max(points)//2

    # for pair in points:
    #     # i1,i2,j1,j2,k1,k2 = pair
    #     i1,j1,k1,i2,j2,k2 = tuple(pair)
    #     ind = int((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
    #     try:
    #         val[ind].append(delta_new[i1, j1, k1] * delta_new[i2, j2, k2])
    #     except KeyError:
    #         val[ind] = [delta_new[i1, j1, k1] * delta_new[i2, j2, k2]]

# #    ####### Computing correlation WITH CIRCLES # NE PAS TOUT CALCULER
    # radius = 10
    # val = {}
    # points = np.random.randint(radius+1,nc-radius-1,size = (10000,3)) #10 000 useful to avoid discriminations betw numbers of low and big distances
    # neighbours  = rP.int_sphere2(radius)
    # # # neighbours  = rP.int_sphere(radius)
    # # print(neighbours.shape)
    # # neighbours  = rP.sphere4(radius,1000)
    # for point in points:
    #     # i1,i2,j1,j2,k1,k2 = pair
    #     i1,j1,k1 = tuple(point)
    #     for neighbour in neighbours:
    #         i2,j2,k2 = tuple(neighbour)
    #         # ind = int((i2) ** 2 + (j2) ** 2+ (k2) ** 2)
    #         ind = (i2) ** 2 + (j2) ** 2+ (k2) ** 2
    #         # if (i1+i2>=0)&(j1+j2>=0)&(k1+k2>= 0)&(i1+i2<=nc)&(j1+j2<=nc)&(k1+k2<=nc):
    #         try:
    #             val[ind].append(delta_new[i1, j1, k1] * delta_new[i1+int(i2), j1+int(j2), k1+int(k2)])
    #         except KeyError:
    #             val[ind] = [delta_new[i1, j1, k1] * delta_new[i1+int(i2), j1+int(j2), k1+int(k2)]]
#     #
#     # # # Computing correlation on threshold-chosen values. # NE PAS TOUT CALCULER
#
#     # val = {}
#     # print(np.std(delta_new))
#     # selected = (delta_new>=0.5)
#     # # plt.imshow((delta_new*selected)[nc//2,:,:])
#     # # plt.show()
#     # b = np.array([[[[i,j,k] for k in range(nc)] for j in range(nc)] for i in range(nc)])
#     # b_selected = b[selected]
#     # print(len(b_selected))
#     # delta_new_selected = delta_new[selected]
#     # print(np.std(delta_new_selected))
#     # facto = np.std(delta_new)/np.std(delta_new_selected)
#     # delta_new_selected = delta_new_selected-np.mean(delta_new_selected)
#     # delta_new_selected *= facto
#     # n =1
#     # for i in range(len(b_selected)):
#     #     for j in range(len(b_selected)):
#     #         ind = int(np.sum((np.ravel(b_selected[i,:])-np.ravel(b_selected[j,:]))**2))
#     #         try:
#     #             val[ind].append(delta_new_selected[i] * delta_new_selected[j])
#     #         except KeyError:
#     #             val[ind] = [delta_new_selected[i] * delta_new_selected[j]]
#
# # # # Computing correlation on pdf-chosen values. # NE PAS TOUT CALCULER
#
#     # # c = 1
#     # K = 1
#     # c = -K/np.min(delta_new)
#     # print('c:',c)
#     # print('K:',K)
#
#     # def pdf():
#     #     '''3D probability probability'''
#     #     # h = (delta_new>=-1)*(1+delta_new) #method 1
#     #     # h = delta_new-np.min(delta_new) #method 2)
#     #     h = K+c*delta_new #method 3
#     #     h= h/np.sum(h)
#     #     return h
#
#     # def rejection_method(n,PDf = pdf()):
#     #     '''choose n points with rejection sampling method for a given pdf'''
#     #     M = np.max(PDf)
#     #     N = int(np.round(n*(np.sum(M-PDf)/np.sum(PDf)))*2*6/np.pi) #because many points go in the bin + we points out of the sphere+points twice-drew
#     #     U = np.round((nc-1)*st.uniform().rvs(size=(N,3))).astype(int)
#     #     H = M*st.uniform().rvs(size=N)
#     #     selection = (PDf[U[:,0],U[:,1],U[:,2]]>=H)
#     #     Uok = U[selection,:]
#     #     sphereTruth = (np.linalg.norm(Uok-nc//2,axis = 1)<=nc//2)
#     #     Uok = Uok[sphereTruth,:]
#     #     indexes = sorted(np.unique(Uok,axis = 0, return_index=True)[1])
#     #     Uok = Uok[indexes,:] # better than just np.unique because np.unique sort values and create bias for the following selection
#     #     return Uok[:np.min([len(Uok),n]),:]
#
#     # b_selected = rejection_method(1000)
#     # delta_new_selected = delta_new[b_selected[:,0],b_selected[:,1],b_selected[:,2]]
#     # m = np.mean(delta_new_selected)
#     # mtheo = c*np.std(delta_new)**2
#     # norm = np.sqrt(K-c*m)
#     # # normvraie =  np.sqrt(K-c**2*np.std(delta_new))
#     # delta_new_selected -= m
#     # # delta_new_selected/=norm
#     # print('mean:',m)
#     # print('mean theoretical:', mtheo)
#     # print('norm:',norm)
#     # # print('norm vraie:',normvraie)
#
#     # print(np.std(delta_new))
#     # print(np.mean(delta_new_selected))
#     # val = {}
#
#     # n =1
#     # for i in range(len(b_selected)):
#     #     for j in range(len(b_selected)):
#     #         ind = int(np.sum((np.ravel(b_selected[i,:])-np.ravel(b_selected[j,:]))**2))
#     #         try:
#     #             val[ind].append(delta_new_selected[i] * delta_new_selected[j])
#     #         except KeyError:
#     #             val[ind] = [delta_new_selected[i] * delta_new_selected[j]]
#


    # val = {}
    # points = np.random.randint(0,nc/2,size = (10000000,6))
    # # normsup = 15#nc//2
    # # points =rP.randomPoints(1, normsup ,100,n = 10000)
    # # points = points+nc//2-np.max(points)//2

    # for pair in points:
    #     # i1,i2,j1,j2,k1,k2 = pair
    #     i1,j1,k1,i2,j2,k2 = tuple(pair)
    #     ind = int((i1 - i2) ** 2 + (j1 - j2) ** 2+ (k1 - k2) ** 2)
    #     try:
    #         val[ind].append(delta_new[i1, j1, k1] * delta_new[i2, j2, k2])
    #     except KeyError:
    #         val[ind] = [delta_new[i1, j1, k1] * delta_new[i2, j2, k2]]

#### Random points and pairs then :
    # a,b = 20,220
    # BinStep = 10
    # val = {}
    # points = np.random.uniform(nc//2-b/(2*dx),nc//2+b/(2*dx),size = (10000,3))
    # # # normsup = 15#nc//2
    # # # points =rP.randomPoints(1, normsup ,100,n = 10000)
    # # # points = points+nc//2-np.max(points)//2
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


#Binning classique :
    # print('...Binning...')
    # X = []
    # Xsi = []
    # for key in val.keys():
    #     if len(val[key]) > 0:#nc**2/2:
    #         if key*dx**2<220**2 and key*dx**2>15**2 : #do not count over 215 Mpc and under 10Mpc (non-linear regime)
    #         # if key in [k**2 for k in range(15)]:
    #             # # # # Xsi = Xsi + val[key]
    #             # # # # X = X + [np.sqrt(key)*dx]*len(val[key])
    #             Xsi.append(np.mean(val[key])) #biased correlation estimation,
    #             print(key,np.sqrt(key) * dx,len(val[key]),np.mean(val[key]))
    #                ### if method 3 only:
    #             # Xsi.append(np.mean(val[key])/c**2)
    #             X.append(np.sqrt(key) * dx)


    if saveCorr == -1:
            ######### Non-biased Corr evualation
        a,b = 20,220 #Mpc
        BinStep = 0.5 #pixels (x20 Mpc)
        dr = BinStep*dx #Mpc
        bins = np.array(np.linspace(a,b,int((b-a)/dr)+1)) 
        N = 4000      
        ### I take a sub-cube with 10 times more 
        n = np.power(10*N,1/3)
        points = np.random.uniform(nc//2-n/2,nc//2+n/2,size = (N,3)) #pixels

        distancesMatrix = sp.distance_matrix(points,points,2)*dx #Mpc
        Delta = np.array([delta_new[points[:,0].astype(int),points[:,1].astype(int),points[:,2].astype(int)]]).T
        xsis = np.kron(Delta,Delta.T)

        Xsi = []
        dispersion = []

        for r in bins:
            xsi, dispersion = XsiEval(r,Dist = distancesMatrix,xsis = xsis,dr = dr)
            Xsi.append(xsi)
        # dispersion.append(disp) # this is not a std, but the dispersion of deltadelta products when taking the mean for xsi

        ##### plot final binning
        plt.scatter(bins,Xsi,linewidths=1,color = 'blue',marker = '+')
        # axs[1,1].errorbar(bins,Xsi, yerr=stdbins,ecolor= 'red')

        # ##### Classic binning
        # plt.scatter(X,Xsi,linewidths=0.5,color = 'blue',marker = '+', alpha=0.6)
        # nbins = 10
        # a,b = 20,220
        # step = (b-a)/nbins
        # centerbins = np.linspace(a,b,nbins+1)
        # print(centerbins)
        # bins = []
        # stdbins = []
        # for i in range(len(centerbins)):
        #     tempbin = []
        #     for j in range(len(X)):
        #         if X[j]>= centerbins[i]-step/2 and X[j]< centerbins[i]+step/2:
        #             tempbin.append(Xsi[j])
        #     bins.append(np.mean(tempbin))
        #     stdbins.append(np.std(tempbin))
        # print('Plotting correlation...')
        # # print(stdbins)
        # axs[1,1].errorbar(centerbins,bins, yerr=stdbins,ecolor= 'red')

    # Reference correlation
        # xref, yref = readtxt('xsi.txt')
        # xref,yref = np.array(xref), np.array(yref)
        # selection = (xref>=10)&(xref<=230)
        # xref,yref = xref[selection], yref[selection]
        # axs[1,1].plot(xref, yref,color = 'red')
        # axs[1,1].semilogx()
        # axs[1,1].semilogy()
        #

        # Analytical fourier⁻1
        # f = lambda k,x : initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
        xr = np.linspace(15,230,50)
        # y = [intg.quad(lambda k : f(k,x), 0, np.inf,limit=100)[0] for x in xr]
        # y = integralXsi(xr,cosmo.Cosmology(sigma_8=np.std(delta_new)))
        # axs[1,1].plot(xr, y,color = 'black',linestyle = '--')
        y = integralXsi(xr,cosmo.Cosmology())
        axs[1,1].plot(xr, y,color = 'red',linestyle = '--')
        
        # axs[1,1].set_xlim([15,220])
        # axs[1,1].set_ylim([-0.025,0.25])
        axs[1,1].legend(['Theoretical','Box (binned)'])
        axs[1,1].set_title('Space correlation function '+r'$\xi$')
        axs[1,1].set_ylabel(r'$\xi(r)$')
        axs[1,1].set_xlabel(r'$r$'+' '+r'$(Mpc)$')
        plt.tight_layout()
        plt.show()
    else:
        delta_new.tofile('heavy files/box'+str(saveCorr)+'nc'+str(int(nc))+'dx'+str(int(dx)))
        #MonteCarlo index = save
        # np.savetxt('heavy files/binsCorrBox'+str(saveCorr)+'.txt',X)
        # np.savetxt('heavy files/CorrBox'+str(saveCorr)+'.txt',Xsi)
        # np.savetxt('heavy files/stdCorrBox',np.ravel(stdbins))




# #==================================
# if __name__ == "__main__":
# #==================================

#     main()

    
    #ref
    # xref, yref = readtxt('xsi.txt')
    # xref,yref = np.array(xref), np.array(yref)
    # selection = (xref>=15)&(xref<=200)
    # xref,yref = xref[selection], yref[selection]
    # yref = 1+yref
    # plt.loglog(xref, yref,color = 'red')

    # #computed
    # x = np.array(np.loadtxt('heavy files/binsCorr3'))
    # y = np.array(np.loadtxt('heavy files/Corr3'))
    # y = 1+y
    # # plt.semilogx()
    # # plt.semilogy()
    # std = np.loadtxt('heavy files/stdCorr3')
    # plt.errorbar(x,y,yerr=(std))

    # plt.xlabel('Radial Distance (Mpc)')
    # plt.ylabel(r'$ 1+\xi(r) $')
    # plt.legend(['Reference', 'Box'])

    # plt.show()

##################### Monte-Carlo, std on Correlation #############################
#FIRst EVALUATE xsis etc
for i in range(0,20):
    print('Iteration ', str(i))
    main(saveCorr = i)

# Then compute uncertainties
# XSIs = []
# for i in range(10):
#     XSIs.append(np.loadtxt('heavy files/CorrBox'+str(i)+'.txt'))

# np.savetxt('heavy files/XSISCorrBox.txt',XSIs)
# Cov = np.cov(np.array(XSIs).T)
# np.savetxt('heavy files/stdCorrBox.txt',np.sqrt(np.diag(Cov)))

# #Then plot
# #ref
# xref, yref = readtxt('xsi.txt')
# xref,yref = np.array(xref), np.array(yref)
# selection = (xref>=15)&(xref<=220)
# xref,yref = xref[selection], yref[selection]
# xref = np.linspace(15,220,100)
# yref = integralXsi(xref,cosmo.Cosmology())
# plt.plot(xref, yref,color = 'red')

#     #computed
# x = np.array(np.loadtxt('heavy files/binsCorrBox0.txt'))
# y = np.array(np.loadtxt('heavy files/CorrBox0.txt'))
# std = np.loadtxt('heavy files/stdCorrBox.txt')
# plt.errorbar(x,y,yerr=(std),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
# plt.scatter(x,y,color = 'blue',marker = '+',linewidths = 0.7)
# plt.xlabel('Radial Distance (Mpc)')
# plt.ylabel(r'$\xi(r) $')
# plt.legend(['Reference', 'Box'])
# plt.show()


>>>>>>> temp
