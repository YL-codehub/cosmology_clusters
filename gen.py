import numpy as np
import matplotlib.pyplot as plt
import csv
import randPoints as rP
import scipy.stats as st

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
    with open('heavy files/'+file, newline='\n') as csvfile1:
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


    nc = 256  # define how many cells your box has
    boxlen = nc*20       # define length of box (Mpc)
    dx = boxlen/nc          # get size of a cell (Mpc), 20Mpc gives ~ 8h^-1 Mpc sphere

    # get overdensity field
    delta = np.random.normal(0, 1, size=(nc, nc, nc))
    print('ffting...')
    delta_k = np.fft.rfftn(delta)/(nc**(3/2)) #equivalent to norm = "ortho")

    # get 3d array of index integer distances to k = (0, 0, 0), ie compute k values' grid in Fourier space
    print('Spectrum incoming...')
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist #²
    dist_z *= dist_z #²
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

    dk = 2*np.pi/boxlen
    kMpc = dist_3d*dk #Mpc^-1

    # Compute spectrum
    P_BKKS = initial_Power_Spectrum_BKKS(kMpc)
    sqP_BKKS = np.sqrt(initial_Power_Spectrum_BKKS(kMpc))
    # print(P_BKKS)
    Spectrum = np.multiply(delta_k,sqP_BKKS)

    # print(Spectrum)

    # Back to real space
    print('inverse ffting...')
    delta_new = np.fft.irfftn(Spectrum, norm = "ortho")*(1/dx)**(3/2) # converting sqrt(P(k))**3 (Mpc^3/2) to cells^3/2
    
    ## Save calculus
    delta_new.tofile('heavy files/boxnc'+str(int(nc))+'dx'+str(int(dx)))
    # delta_new2 = np.fromfile('box nc='+str(nc)+' and dx = '+str(dx))
    # delta_new2 = np.reshape(delta_new2,(nc,nc,nc))

    # plot overdensity constrast
    print('Plotting contrasts slices...')
    fig, axs = plt.subplots(2, 2)
    axs[0,0].imshow(delta[nc//2,:,:])
    axs[0,0].set_title('Initial white noise')


    axs[1, 0].imshow(delta_new[nc // 2, :, :])
    axs[1, 0].set_title(r'Overensity contrast at z = 0, $\sigma_8 =$ '+str(np.std(delta_new).round(2)))

    # Check Spectrum (and so the units !)
    print('Plotting Spectrum...')
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
    # plt.show()

    print('Computing correlation...')
    print('...Evaluating...')
    # # # Computing correlation. # NE PAS TOUT CALCULER
    # val = {}
    # n =1
    # for i1 in range(nc):
    #     for i2 in range(nc):
    #         for j1 in range(nc):
    #             for j2 in range(nc):
    #                 for k1 in range(nc):
    #                     for k2 in range(nc):
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

   ####### Computing correlation WITH CIRCLES # NE PAS TOUT CALCULER
    radius = 11
    val = {}
    points = np.random.randint(radius+1,nc-radius,size = (10000,3))
    # normsup = 15#nc//2
    # points =rP.randomPoints(1, normsup ,100,n = 10000)
    # points = points+nc//2-np.max(points)//2
    neighbours  = rP.int_sphere(radius)
    # print(neighbours.shape)
    for point in points:
        # i1,i2,j1,j2,k1,k2 = pair
        i1,j1,k1 = tuple(point)
        for neighbour in neighbours:
            i2,j2,k2 = tuple(neighbour)
            ind = int((i2) ** 2 + (j2) ** 2+ (k2) ** 2)
            # if (i1+i2>=0)&(j1+j2>=0)&(k1+k2>= 0)&(i1+i2<=nc)&(j1+j2<=nc)&(k1+k2<=nc):
            try:
                val[ind].append(delta_new[i1, j1, k1] * delta_new[i1+i2, j1+j2, k1+k2])
            except KeyError:
                val[ind] = [delta_new[i1, j1, k1] * delta_new[i1+i2, j1+j2, k1+k2]]
        
    # # # Computing correlation on threshold-chosen values. # NE PAS TOUT CALCULER
 
    # val = {}
    # print(np.std(delta_new))
    # selected = (delta_new>=0.5)
    # # plt.imshow((delta_new*selected)[nc//2,:,:])
    # # plt.show()
    # b = np.array([[[[i,j,k] for k in range(nc)] for j in range(nc)] for i in range(nc)])
    # b_selected = b[selected]
    # print(len(b_selected))
    # delta_new_selected = delta_new[selected]
    # print(np.std(delta_new_selected))
    # facto = np.std(delta_new)/np.std(delta_new_selected)
    # delta_new_selected = delta_new_selected-np.mean(delta_new_selected)
    # delta_new_selected *= facto
    # n =1
    # for i in range(len(b_selected)):
    #     for j in range(len(b_selected)):
    #         ind = int(np.sum((np.ravel(b_selected[i,:])-np.ravel(b_selected[j,:]))**2))
    #         try:
    #             val[ind].append(delta_new_selected[i] * delta_new_selected[j])
    #         except KeyError:
    #             val[ind] = [delta_new_selected[i] * delta_new_selected[j]]

# # # Computing correlation on pdf-chosen values. # NE PAS TOUT CALCULER
    
    # # c = 1
    # K = 1
    # c = -K/np.min(delta_new)
    # print('c:',c)
    # print('K:',K)

    # def pdf():
    #     '''3D probability probability'''
    #     # h = (delta_new>=-1)*(1+delta_new) #method 1
    #     # h = delta_new-np.min(delta_new) #method 2)
    #     h = K+c*delta_new #method 3
    #     h= h/np.sum(h)
    #     return h

    # def rejection_method(n,PDf = pdf()):
    #     '''choose n points with rejection sampling method for a given pdf'''
    #     M = np.max(PDf)
    #     N = int(np.round(n*(np.sum(M-PDf)/np.sum(PDf)))*2*6/np.pi) #because many points go in the bin + we points out of the sphere+points twice-drew
    #     U = np.round((nc-1)*st.uniform().rvs(size=(N,3))).astype(int)
    #     H = M*st.uniform().rvs(size=N) 
    #     selection = (PDf[U[:,0],U[:,1],U[:,2]]>=H)
    #     Uok = U[selection,:]
    #     sphereTruth = (np.linalg.norm(Uok-nc//2,axis = 1)<=nc//2)
    #     Uok = Uok[sphereTruth,:]
    #     indexes = sorted(np.unique(Uok,axis = 0, return_index=True)[1])
    #     Uok = Uok[indexes,:] # better than just np.unique because np.unique sort values and create bias for the following selection
    #     return Uok[:np.min([len(Uok),n]),:]

    # b_selected = rejection_method(1000)
    # delta_new_selected = delta_new[b_selected[:,0],b_selected[:,1],b_selected[:,2]]
    # m = np.mean(delta_new_selected)
    # mtheo = c*np.std(delta_new)**2
    # norm = np.sqrt(K-c*m)
    # # normvraie =  np.sqrt(K-c**2*np.std(delta_new))
    # delta_new_selected -= m
    # # delta_new_selected/=norm
    # print('mean:',m)
    # print('mean theoretical:', mtheo)
    # print('norm:',norm)
    # # print('norm vraie:',normvraie)

    # print(np.std(delta_new))
    # print(np.mean(delta_new_selected))
    # val = {}
    
    # n =1
    # for i in range(len(b_selected)):
    #     for j in range(len(b_selected)):
    #         ind = int(np.sum((np.ravel(b_selected[i,:])-np.ravel(b_selected[j,:]))**2))
    #         try:
    #             val[ind].append(delta_new_selected[i] * delta_new_selected[j])
    #         except KeyError:
    #             val[ind] = [delta_new_selected[i] * delta_new_selected[j]]


    print('...Binning...')   
    X = []
    Xsi = []
    for key in val.keys():
        if len(val[key]) > 0:#nc**2/2:
            if key*dx**2<210**2 and key*dx**2>10**2: #do not count over 200 Mpc and under 20Mpc (non-linear regime)
                # # # # Xsi = Xsi + val[key]
                # # # # X = X + [np.sqrt(key)*dx]*len(val[key])
                Xsi.append(np.mean(val[key])) #biased correlation estimation,
                   ### if method 3 only:
                # Xsi.append(np.mean(val[key])/c**2)
                X.append(np.sqrt(key) * dx)
                 
                
    # #     # Xsi.append(np.sum(val[key])/(len(val[key])-np.sqrt(key))) #unbiased correlation estimation, estimateur est rarement utilisé car sa variance est très élevée pour les valeurs de k proches de N, et en général moins bon que le cas biaisé
    # # # # print(X,Xsi)
    plt.scatter(X,Xsi,linewidths=0.5,color = 'blue',marker = '+', alpha=0.6)
    # # # # axs[1,1].set_ylabel('Correlation function estimation')
    # # # # axs[1,1].set_xlabel('Radial distance (Mpc)')

    # print(np.max(X))
    # # Correlation bins
    print(len(val.keys()))
    nbins = 20
    # beginbins = np.linspace(0,np.max(X),nbins)
    beginbins = np.linspace(1,210,nbins)
    bins = []
    stdbins = []
    for i in range(nbins-1):
        tempbin = []
        for j in range(len(X)):
            if X[j]>= beginbins[i] and X[j]< beginbins[i+1]:
                tempbin.append(Xsi[j])
        bins.append(np.mean(tempbin))
        stdbins.append(np.std(tempbin))
    print('Plotting correlation...')
    # print(stdbins)
    axs[1,1].errorbar(beginbins[:-1]+0.5*np.max(X)/nbins,bins, yerr=stdbins,ecolor= 'red')

# Reference correlation
    xref, yref = readtxt('xsi.txt')
    xref,yref = np.array(xref), np.array(yref)
    selection = (xref>=10)&(xref<=210)
    xref,yref = xref[selection], yref[selection]
    axs[1,1].plot(xref, yref,color = 'red')
    # axs[1,1].semilogx()
    # axs[1,1].semilogy()
    axs[1,1].legend(['Ref','Mine (Points)','Mine (bins)'])
    # 

    # Analytical fourier⁻1
    # f = lambda k,x : initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
    # xr = np.linspace(20,200,50)
    # y = [intg.quad(lambda k : f(k,x), 0, np.inf,limit=100)[0] for x in xr]
    # axs[1,1].plot(xr, y,color = 'red',linestyle = '--')
    
    # plt.xlim([20,100])
    plt.show()

    np.savetxt('heavy files/binsCorr3',beginbins[:-1]+0.5*np.max(X)/nbins)
    np.savetxt('heavy files/Corr3',np.ravel(bins))
    np.savetxt('heavy files/stdCorr3',np.ravel(stdbins))




#==================================
if __name__ == "__main__":
#==================================

    main()

    
    #ref
    xref, yref = readtxt('xsi.txt')
    xref,yref = np.array(xref), np.array(yref)
    selection = (xref>=15)&(xref<=200)
    xref,yref = xref[selection], yref[selection]
    yref = 1+yref
    plt.loglog(xref, yref,color = 'red')

    #computed
    x = np.array(np.loadtxt('heavy files/binsCorr3'))
    y = np.array(np.loadtxt('heavy files/Corr3'))
    y = 1+y
    # plt.semilogx()
    # plt.semilogy()
    std = np.loadtxt('heavy files/stdCorr3')
    plt.errorbar(x,y,yerr=(std))

    plt.xlabel('Radial Distance (Mpc)')
    plt.ylabel(r'$ 1+\xi(r) $')
    plt.legend(['Reference', 'Box'])

    plt.show()