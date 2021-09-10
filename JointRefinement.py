from numpy.linalg.linalg import norm
import Cosmological_tools as cosmo
import scipy.optimize as opt
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.interpolate 
import math as m


def readtxt(file):
    '''To read reference files/plot, with two columns'''
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


def integralXsi(R,univ,a = 1e-7, b= 1e3,n = 100000):
    '''To compute efficiently the correlation xsi at radial distances stored in R and depending in the cosmology. Other parameters are for the discretization of the integration'''
    K = np.array([np.linspace(a, b, n)])
    dK = (b - a) / n
    X = np.array([R]).T
    prod = np.kron(K, X)
    F = univ.initial_Power_Spectrum_BKKS(K, mode='np') * np.sin(K * X) / (K * X) * K ** 2 / (2 * np.pi ** 2)
    xsi_model = np.sum(F, axis=1) * dK
    return(xsi_model)


def plot_likelihood(corrfilename,countsfilename,O,S,Minf = 1.6e14,Msup = 1e16, zinf = 0,zsup = 0.6118479756853465,dlogM = 0.1,dz = 0.1, mode = 'multi'):
    '''Plot likelihood of the joint (multi mode) refinement of both the counts and the correlation. Possibility to refine only one of them depending on the mode.'''
    universe = cosmo.Cosmology()

    #### LOAD DATA, from the same catalog !!!
    # Load Correlation Data
    r = np.loadtxt('heavy files/binsCorr'+corrfilename+'.txt')
    xsi = np.array([np.loadtxt('heavy files/Corr'+corrfilename+'.txt')])
    std = np.loadtxt('heavy files/stdCorrBig0.txt')
    Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
    np.fill_diagonal(Sigma,std**2)
    invSig = np.linalg.inv(Sigma)

    #Load counts 
    counts = np.loadtxt('heavy files/Bins_0.1_0.1'+countsfilename+'.txt')
    a,b = m.log(Minf, 10), m.log(Msup, 10)
    Mmin = np.power(10, np.linspace(a, b, int((b - a) / dlogM) + 1))
    Mmax = Mmin * 10 ** dlogM
    zmin = np.linspace(zinf, zsup, int((zsup - zinf) / dz) + 1)
    zmax = zmin + dz
    ###############
    

    def loglikelihood(parameters):
        print(parameters)
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        # Correlation part :
        if mode != 'counts':
            xsi_model = integralXsi(r,universe)
        # Counts part :
        res = 0
        if mode != 'corr':
            for i in range(len(counts)):
                for j in range(len(counts[0])):
                    mean_poisson = universe.expected_Counts(Mmin[j], Mmax[j], zmin[i], zmax[i],rad2 = 4*np.pi)
                    res += -0.5*(mean_poisson-counts[i,j])**2/mean_poisson  #diagonal Gaussian cov
        # sum or not
        if mode == 'counts':
            return(res)
        if mode == 'corr':
            return -np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]
        return -np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0] + res
        
    Z = np.array([[loglikelihood([o,s]) for s in S] for o in O])

    a = np.argmax(Z)
    print('Omega_m :', O[a % len(S)])
    print('Sigma :', S[a // len(S)])
        #
    X, Y = np.meshgrid(O, S)
    Z = np.exp(Z - Z.max())

    Z = Z / Z.sum()
    #     # #
    t = np.linspace(0, Z.max(), 1000)
    integral = ((Z >= t[:, None, None]) * Z).sum(axis=(1, 2))
    f = scipy.interpolate.interp1d(integral, t)
    t_contours = f(np.array([0.95, 0.68]))

    ax1 = plt.subplot(121)
    # ax1.contourf(X, Y, Z)
    ax1.scatter(O[a % len(S)], S[a // len(S)])
    ax1.contour(X, Y, Z, t_contours,colors = ['red','blue'],alpha = 0.5)
    ax1.set_xlabel(r'$\Omega_m$')
    ax1.set_ylabel(r'$\sigma_8$')

    ## Plot effective correlation versus data
    ax2 = plt.subplot(122)
    #ref
    xref = np.linspace(15,220,50)
    yref = integralXsi(xref, cosmo.Cosmology())
    ax2.plot(xref, yref,color = 'black',linewidth = 1)
    #data
    ax2.errorbar(r,xsi.T, yerr=std,fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
    # refined
    universe = cosmo.Cosmology(Omega_m=O[a % len(S)],Omega_v=1-O[a % len(S)], sigma_8=S[a // len(S)])
    xsi_model = integralXsi(xref,universe)
    ax2.scatter(r,xsi[0], color = 'blue',marker = '+')
    ax2.plot(xref,xsi_model.T, color = 'blue',linestyle = '--',linewidth = 1)

    ax2.legend(['Theoretical','Refined','Data'])
    ax2.set_xlabel('Radial distance (Mpc)')
    ax2.set_ylabel('Correlation function')
    ax2.set_xlim([15,225])
    ax2.set_ylim([-0.025,0.20])
    plt.tight_layout()
    plt.show()


def refineMax(corrfilename,countsfilename,Minf = 1.6e14,Msup = 1e16, zinf = 0,zsup = 0.6118479756853465,dlogM = 0.1,dz = 0.1, mode = 'multi', plot = False):
    '''Joint (multi mode) refinement of both the counts and the correlation. Possibility to refine only one of them depending on the mode.'''
    universe = cosmo.Cosmology()
    #### LOAD DATA, from the same catalog !!!
    # Load Correlation Data
    r = np.loadtxt('heavy files/binsCorr'+corrfilename+'.txt')
    xsi = np.array([np.loadtxt('heavy files/Corr'+corrfilename+'.txt')])
    # xsi = np.array([integralXsi(r,cosmo.Cosmology())])
    std = np.loadtxt('heavy files/stdCorrBig0.txt')
    # Sigma = np.eye(len(xsi[0]))
    Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
    np.fill_diagonal(Sigma,std**2)
    invSig = np.linalg.inv(Sigma)

    #Load counts 
    counts = np.loadtxt('heavy files/Bins_0.1_0.1'+countsfilename+'.txt')
    a,b = m.log(Minf, 10), m.log(Msup, 10)
    Mmin = np.power(10, np.linspace(a, b, int((b - a) / dlogM) + 1))
    Mmax = Mmin * 10 ** dlogM
    zmin = np.linspace(zinf, zsup, int((zsup - zinf) / dz) + 1)
    zmax = zmin + dz
    ###############
    

    def loglikelihood(parameters):
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        # Correlation part :
        if mode != 'counts':
            xsi_model = integralXsi(r,universe)
        # Counts part :
        res = 0
        if mode != 'corr':
            for i in range(len(counts)):
                for j in range(len(counts[0])):
                    mean_poisson = universe.expected_Counts(Mmin[j], Mmax[j], zmin[i], zmax[i],rad2 = 4*np.pi)
                    res += -0.5*(mean_poisson-counts[i,j])**2/mean_poisson  #diagonal cov
        # sum or not
        if mode == 'counts':
            return(-res)
        if mode == 'corr':
            return +np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]
        return +np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0] -res
      
    sol = opt.minimize(loglikelihood, [0.3,0.8],options={ 'disp': True}, bounds = ((0,1),(0,2))).x

    if plot:
        # ref versus data
        ax2 = plt.subplot()
        #ref
        xref = np.linspace(15,220,50)
        yref = integralXsi(xref,cosmo.Cosmology())
        ax2.plot(xref, yref,color = 'black',linewidth = 1)
        #data
        ax2.errorbar(r,xsi.T, yerr=std,fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
        # refined
        universe = cosmo.Cosmology(Omega_m=sol[0],Omega_v=1-sol[0], sigma_8=sol[1])
        
        xsi_model = integralXsi(xref,universe)
        ax2.scatter(r,xsi[0], color = 'blue',marker = '+')
        ax2.plot(xref,xsi_model.T, color = 'blue',linestyle = '--',linewidth = 1)

        ax2.legend(['Theoretical','Refined : '+str(list(np.round(sol,3))),'Data'])
        ax2.set_xlabel('Radial distance (Mpc)')
        ax2.set_ylabel('Correlation function')
        ax2.set_xlim([15,225])
        ax2.set_ylim([-0.025,0.20])
        plt.show()

    return(sol) #

# Choose, uncomment and modify :
# plot_likelihood('BigCorrelationmean','BigDoubleCatalog2',np.linspace(0.25,0.35,41),np.linspace(0.75,0.85,41),mode = 'multi')
# print(refineMax('BigCorrelationmean','BigDoubleCatalog2',mode = 'multi',plot = True))

