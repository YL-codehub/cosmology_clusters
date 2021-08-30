from numpy.linalg.linalg import norm
import Cosmological_tools as cosmo
import scipy.optimize as opt
import numpy as np
# import dataCorrelation as dC
import csv
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.interpolate 

# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122)

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


    # Array integral mode :
def integralXsi(R,univ,a = 1e-7, b= 1e3,n = 100000):
    K = np.array([np.linspace(a, b, n)])
    dK = (b - a) / n
    X = np.array([R]).T
    prod = np.kron(K, X)
    F = univ.initial_Power_Spectrum_BKKS(K, mode='np') * np.sin(K * X) / (K * X) * K ** 2 / (2 * np.pi ** 2)
    xsi_model = np.sum(F, axis=1) * dK
    return(xsi_model)

def plot_likelihood(r,xsi,O,S,std,mode = 'diagonal'):
    '''Xsi(r) = fourier^-1{spectrum) on xsi(r) data points 
    (usually coming from Landy and Szaslay estimator).'''
    universe = cosmo.Cosmology()
    xsi = np.array([xsi])
    std = np.array(std)
    if mode == 'diagonal':
        Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
        np.fill_diagonal(Sigma,std**2)
        std = Sigma
    invSig = np.linalg.inv(std)


    # print(integralXsi(r,cosmo.Cosmology(Omega_m=0.3,Omega_v=1-0.3, sigma_8=0.8)))
    def loglikelihood(parameters):
        print(parameters)
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        # f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
        # xsi_model = np.array([[intg.quad(lambda k : f(k,x), 0, np.inf,limit=1000)[0] for x in r]])
        xsi_model = integralXsi(r,universe)
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
    # print(integral)
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
    xref, yref = readtxt('Vérifications fonctions/xsi.txt')
    ax2.plot(xref, yref,color = 'black',linewidth = 1)
    #data
    ax2.errorbar(r,xsi.T, yerr=std.diagonal(),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
    # refined
    universe = cosmo.Cosmology(Omega_m=O[a % len(S)],Omega_v=1-O[a % len(S)], sigma_8=S[a // len(S)])
    # f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
    # xsi_model = np.array([[intg.quad(lambda k : f(k,x), 0, np.inf,limit=1000)[0] for x in r]])
    xsi_model = integralXsi(xref,universe)
    ax2.scatter(r,xsi[0], color = 'blue',marker = '+')
    ax2.plot(xref,xsi_model.T, color = 'blue',linestyle = '--',linewidth = 1)

    # universeRef = cosmo.Cosmology()
    # xsi_ref = integralXsi(r,universeRef)
    # ax2.plot(r, xsi_ref,color = 'red',linewidth = 1) #ok equivalent to theory

    ax2.legend(['Theoretical','Refined','Data'])
    ax2.set_xlabel('Radial distance (Mpc)')
    ax2.set_ylabel('Correlation function')
    ax2.set_xlim([15,225])
    ax2.set_ylim([-0.025,0.20])
    plt.show()


def refineMax(r,xsi,std, plot = False, mode = 'diagonal'):
    universe = cosmo.Cosmology()
    xsi = np.array([xsi])
    std = np.array(std)
    if mode == 'diagonal':
        Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
        np.fill_diagonal(Sigma,std**2)
        std = Sigma
    invSig = np.linalg.inv(std)
    


    # print(integralXsi(r,cosmo.Cosmology(Omega_m=0.3,Omega_v=1-0.3, sigma_8=0.8)))
    def loglikelihood(parameters):
        # print(parameters)
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        # f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
        # xsi_model = np.array([[intg.quad(lambda k : f(k,x), 0, np.inf,limit=1000)[0] for x in r]])
        xsi_model = integralXsi(r,universe)
        return np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]

    # sol = opt.minimize(loglikelihood, [0.3,0.8],options={ 'disp': True}, bounds = ((0,1),(0,2))).x
    sol = opt.minimize(loglikelihood, [0.3,0.8], bounds = ((0,1),(0,2))).x

    if plot:
        # ref versus data
        ax2 = plt.subplot()
        #ref
        xref, yref = readtxt('Vérifications fonctions/xsi.txt')
        ax2.plot(xref, yref,color = 'black',linewidth = 1)
        #data
        ax2.errorbar(r,xsi.T, yerr=std.diagonal(),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
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
# x = np.loadtxt('heavy files/binsCorrBig0.txt')
# # y = np.loadtxt('heavy files/CorrBig0.txt')
# y = integralXsi(x,cosmo.Cosmology())
# std = np.loadtxt('heavy files/stdCorrBig0.txt')
# # XsisMC = np.loadtxt('heavy files/XSIsBig.txt').T
# # std = np.cov(XsisMC)
# plot_likelihood(x, y, np.linspace(0.20,0.40,31),np.linspace(0.7,0.9,31),std)
# # print(refineMax(x,y,std,plot=True,mode = 'non-diagonal'))
# # stddiag = np.loadtxt('heavy files/stdCorrBig0.txt')
# # print(refineMax(x,y,stddiag,plot=True))

#######################
# Tests 1 réalisation #
#######################
# for i in range(10):
#     print('iteration: ', i)
#     x = np.loadtxt('heavy files/binsCorrMC'+str(i)+'.txt')
#     y = np.loadtxt('heavy files/CorrMC'+str(i)+'.txt')
#     std = np.loadtxt('heavy files/stdCorrMC'+str(i)+'.txt')
#     plot_likelihood(x, y, np.linspace(0.05,0.65,20),np.linspace(0.5,1.1,20),std)
# # plt.show()

###########################
# Refine all MC and store #
# ###########################
Sols = []
for i in range(0,40):
    try:
        print('Iteration '+str(i)+' :')
        x = np.loadtxt('heavy files/CorrbinsBigCorrelation'+str(i)+'.txt')
        y = np.loadtxt('heavy files/CorrBigCorrelation'+str(i)+'.txt')  
        std = np.loadtxt('heavy files/stdCorrBig0.txt') # same uncertainties for all
        sol = refineMax(x,y,std,plot= False )
        print(sol)
        Sols.append(sol)
        print('------------------')
    except:
        continue
np.savetxt('heavy files/optiBig2.txt',Sols)




