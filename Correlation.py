from numpy.linalg.linalg import norm
import Cosmological_tools as cosmo
import scipy.optimize as opt
import numpy as np
# import dataCorrelation as dC
import csv
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.interpolate 

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


def plot_likelihood(r,xsi,O,S,std):
    '''Xsi(r) = fourier^-1{spectrum) on xsi(r) data points 
    (usually coming from Landy and Szaslay estimator).'''
    universe = cosmo.Cosmology()
    xsi = np.array([xsi])
    std = np.array(std)
    # Sigma = np.eye(len(xsi[0]))
    Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
    np.fill_diagonal(Sigma,std**2)
    invSig = np.linalg.inv(Sigma)

    def loglikelihood(parameters):
        print(parameters)
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
        xsi_model = np.array([[intg.quad(lambda k : f(k,x), 0, np.inf,limit=1000)[0] for x in r]])
        return -np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]
        
    Z = np.array([[loglikelihood([o,s]) for s in S] for o in O])
        # np.savetxt('countsresultZ',Z)
        # np.savetxt('countsresultO',O)
        # np.savetxt('countsresultS',S)

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
    xref, yref = readtxt('xsi.txt')
    ax2.plot(xref, yref,color = 'black',linewidth = 1)
    #data 
    ax2.errorbar(r,xsi.T, yerr=std,fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
    # refined
    universe.Om =  O[a % len(S)]
    universe.sigma8 = S[a // len(S)]
    universe.update()
    f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
    xsi_model = np.array([[intg.quad(lambda k : f(k,x), 0, np.inf,limit=1000)[0] for x in r]])
    ax2.scatter(r,xsi[0], color = 'blue',marker = '+')
    ax2.plot(r,xsi_model[0], color = 'blue',linestyle = '--',linewidth = 1)
    ax2.legend(['Theoretical','Refined','Data'])
    ax2.set_xlabel('Radial distance (Mpc)')
    ax2.set_ylabel('Correlation function')
    ax2.set_xlim([15,225])
    ax2.set_ylim([-0.025,0.2])
    plt.show()


###########################################################
## Verifications :
#####################theoretical correlation###############

# universe = cosmo.Cosmology()
# # Ref :


# # Analytical fourier⁻1
# f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
# y = [intg.quad(lambda k : f(k,x), 0, np.inf,limit=100)[0] for x in xref]

# plt.plot(xref, y,color = 'blue',linestyle = '--')
# plt.legend(['Ref','Mine'])
# plt.show()

#####################likelihood plot###############

# plot_likelihood(xref, yref, np.linspace(0.2,0.4,5),np.linspace(0.7,0.9,5))

##################
x = np.loadtxt('heavy files/binsCorr.txt')
y = np.loadtxt('heavy files/Corr.txt')
# x, y = readtxt('xsi.txt') #testing likelihood algorithm
std = np.loadtxt('heavy files/stdCorr.txt')
# std = [0.01]*len(y) #testing likelihood algorithm
plot_likelihood(x, y, np.linspace(0.2,0.4,7),np.linspace(0.7,0.9,7),std)