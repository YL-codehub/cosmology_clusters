from numpy.linalg.linalg import norm
import Cosmological_tools as cosmo
import scipy.optimize as opt
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.interpolate 


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

def plot_likelihood(r,xsi,O,S,std,mode = 'diagonal'):
    '''Refinement of Xsi(r) = fourier^-1{spectrum) on xsi(r) data points 
    (usually coming from Landy and Szaslay estimator by dataCorrelation.py).'''
    universe = cosmo.Cosmology()
    xsi = np.array([xsi])
    std = np.array(std)
    if mode == 'diagonal':
        Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
        np.fill_diagonal(Sigma,std**2)
        std = Sigma
    invSig = np.linalg.inv(std)

    def loglikelihood(parameters):
        print(parameters)
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        xsi_model = integralXsi(r,universe)
        return -np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]
        
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
    ax1.scatter(O[a % len(S)], S[a // len(S)], marker = '+')
    ax1.contour(X, Y, Z, t_contours,colors = ['red','blue'],alpha = 0.5)
    ax1.set_xlabel(r'$\Omega_m$')
    ax1.set_ylabel(r'$\sigma_8$')

    ## Plot effective correlation versus data
    ax2 = plt.subplot(122)
    #ref
    xref = np.linspace(15,230,50)
    yref = integralXsi(xref,cosmo.Cosmology())
    ax2.plot(xref, yref,color = 'black',linewidth = 1)
    #data
    ax2.errorbar(r,xsi.T, yerr=np.sqrt(std.diagonal()),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
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


def refineMax(r,xsi,std, plot = False, mode = 'diagonal'):
    universe = cosmo.Cosmology()
    xsi = np.array([xsi])
    std = np.array(std)
    if mode == 'diagonal':
        Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
        np.fill_diagonal(Sigma,std**2)
        std = Sigma
    invSig = np.linalg.inv(std)
    
    def loglikelihood(parameters):
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        xsi_model = integralXsi(r,universe)
        return np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]

    sol = opt.minimize(loglikelihood, [0.2,0.7],options={ 'disp': True}, bounds = ((0,1),(0,2))).x

    if plot:
        # ref versus data
        ax2 = plt.subplot()
        #ref
        xref = np.linspace(15,230,50)
        yref = integralXsi(xref,cosmo.Cosmology())
        ax2.plot(xref, yref,color = 'black',linewidth = 1)
        #data
        ax2.errorbar(r,xsi.T, yerr=np.sqrt(std.diagonal()),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
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


##################### Report :

###### Box refinement
# x = np.array(np.loadtxt('heavy files/binsCorrBigMC0.txt'))
# y = np.array(np.loadtxt('heavy files/CorrBigMCmean.txt'))
# std = np.array(np.loadtxt('heavy files/stdCorrBigMC.txt'))
# print(refineMax(x,y,std,plot=True))
# plot_likelihood(x, y, np.linspace(0.1,0.6,51),np.linspace(0.6,1.1,51),std)

###########################
# Refine all MC and store #
# # ###########################
# Sols = []
# for i in range(0,20):
#     print('Iteration '+str(i)+' :')
#     x = np.loadtxt('heavy files/binsCorrBigMC0.txt')
#     y = np.loadtxt('heavy files/CorrBigMC'+str(i)+'.txt')  
#     std = np.loadtxt('heavy files/stdCorrBigMC.txt') # same uncertainties for all
#     sol = refineMax(x,y,std,plot= False)
#     print(sol)
#     Sols.append(sol)
#     print('------------------')

# np.savetxt('heavy files/optiBigMC.txt',Sols)

################################ Testing window effects on correlation

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

# r = np.linspace(15,100,50)
# xsi_th = integralXsiMode(r,cosmo.Cosmology())
# xsi_convoluted10 = integralXsiMode(r,cosmo.Cosmology(),mode = 10)
# xsi_convoluted17 = integralXsiMode(r,cosmo.Cosmology(),mode = 10*np.sqrt(3)) #17 = 10*sqrt(3)
# plt.plot(r,xsi_th)
# plt.plot(r,xsi_convoluted10)
# plt.plot(r,xsi_convoluted17)
# plt.title('Theoretical space correlation functions')
# plt.xlabel('Radial distance (Mpc)')
# plt.ylabel('Correlation function '+r'$\xi$')

# # #effective in the box :
# x = np.array(np.loadtxt('heavy files/binsCorrBoxMean.txt'))
# y = np.array(np.loadtxt('heavy files/CorrBoxMean.txt'))
# std = np.loadtxt('heavy files/stdCorrBox.txt')
# plt.errorbar(x,y, yerr=std,fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
# plt.scatter(x,y, color = 'blue',marker = '+')
# plt.legend([r'$\xi$',r'$10$'+'Mpc-top-hat convoluted '+r'$\xi$', r'$10\sqrt{3}$'+'Mpc-top-hat convoluted '+r'$\xi$','Box'])
       
# plt.show()

# plt.legend([r'$\xi$',r'$10$'+'Mpc-top-hat convoluted '+r'$\xi$', r'$10\sqrt{3}$'+'Mpc-top-hat convoluted '+r'$\xi$'])


######### find the best R near data #########:
# number = 'mean'
# # number = 1
# def objective_function(R):
#     r = np.array(np.loadtxt('heavy files/binsCorrBigMC0.txt'))
#     # r = np.delete(r,[25,69,81]) # delete 40Mpc anomaly
#     xsi_data = np.array(np.loadtxt('heavy files/CorrBigMC'+str(number)+'.txt'))
#     # xsi_data = np.delete(xsi_data,[25,69,81]) # delete 40Mpc anomaly
#     std = np.array(np.loadtxt('heavy files/stdCorrBigMC.txt'))
#     # std = np.delete(std,[25,69,81]) # delete 40Mpc anomaly
#     xsi_convolutedR = integralXsiMode(r,cosmo.Cosmology(),mode = R)
#     return np.sum(np.power((xsi_convolutedR-xsi_data)/std,2))

# import scipy.optimize as opt
# res = opt.minimize(objective_function, [10],options={ 'disp': True}, bounds = ((5,16),)).x
# print('R = ',res)

# # plot 
# r = np.linspace(15,175,50)
# xsi_th = integralXsiMode(r,cosmo.Cosmology())
# xsi_convolutedR= integralXsiMode(r,cosmo.Cosmology(),mode = res)
# plt.plot(r,xsi_th, color = 'grey',linewidth = 1)
# plt.plot(r,xsi_convolutedR,color = 'black',linestyle = '--',linewidth = 1)
# plt.title('Theoretical space correlation functions')
# plt.xlabel('Radial distance (Mpc)')
# plt.ylabel('Correlation function '+r'$\xi$')

# #effective in the box :
# x = np.array(np.loadtxt('heavy files/binsCorrBox0.txt'))
# y = np.array(np.loadtxt('heavy files/CorrBox0.txt'))
# std = np.loadtxt('heavy files/stdCorrBox.txt')
# plt.errorbar(x,y, yerr=std,fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
# plt.scatter(x,y, color = 'blue',marker = '+')
# plt.legend([r'$\xi$',str(np.round(res,2)[0])+'Mpc-top-hat convoluted '+r'$\xi$','Box'])
       
# plt.show()


#####################"" Refine with the convoluted model

def refineMaxConvol(r,xsi,std, R,plot = False, mode = 'diagonal'):
    universe = cosmo.Cosmology()
    xsi = np.array([xsi])
    std = np.array(std)
    if mode == 'diagonal':
        Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
        np.fill_diagonal(Sigma,std**2)
        std = Sigma
    invSig = np.linalg.inv(std)
    
    def loglikelihood(parameters):
        # print(parameters)
        universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
        xsi_model = integralXsiMode(r,universe,mode = R)
        return np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]

    sol = opt.minimize(loglikelihood, [0.2,0.7],options={ 'disp': True}, bounds = ((0,1),(0,2))).x

    if plot:
        # ref versus data
        ax2 = plt.subplot()
        #ref
        xref = np.linspace(np.min(r)-5,np.max(r)+5,200)
        yref = integralXsiMode(xref,cosmo.Cosmology(),mode = R)
        ax2.plot(xref, yref,color = 'black',linewidth = 1)
        #data
        ax2.errorbar(r,xsi.T, yerr=np.sqrt(std.diagonal()),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
        # refined
        universe = cosmo.Cosmology(Omega_m=sol[0],Omega_v=1-sol[0], sigma_8=sol[1])
        
        xsi_model = integralXsiMode(xref,universe,mode = R)
        ax2.scatter(r,xsi[0], color = 'blue',marker = '+')
        ax2.plot(xref,xsi_model.T, color = 'blue',linestyle = '--',linewidth = 1)

        ax2.legend(['Theoretical Convoluted','Refined : '+str(list(np.round(sol,3))),'Data'])
        ax2.set_xlabel('Radial distance (Mpc)')
        ax2.set_ylabel('Correlation function')
        ax2.set_xlim([15,225])
        ax2.set_ylim([-0.025,0.20])
        plt.show()

    return(sol) #

# def plot_likelihoodConvol(r,xsi,O,S,std,R,mode = 'diagonal'):
#     '''Xsi(r) = fourier^-1{spectrum) on xsi(r) data points 
#     (usually coming from Landy and Szaslay estimator).'''
#     universe = cosmo.Cosmology()
#     xsi = np.array([xsi])
#     std = np.array(std)
#     if mode == 'diagonal':
#         Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
#         np.fill_diagonal(Sigma,std**2)
#         std = Sigma
#     invSig = np.linalg.inv(std)


#     # print(integralXsi(r,cosmo.Cosmology(Omega_m=0.3,Omega_v=1-0.3, sigma_8=0.8)))
#     def loglikelihood(parameters):
#         print(parameters)
#         universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
#         # f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
#         # xsi_model = np.array([[intg.quad(lambda k : f(k,x), 0, np.inf,limit=1000)[0] for x in r]])
#         xsi_model = integralXsiMode(r,universe,mode = R)
#         return -np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]
        
#     Z = np.array([[loglikelihood([o,s]) for s in S] for o in O])
#         # np.savetxt('countsresultZ',Z)
#         # np.savetxt('countsresultO',O)
#         # np.savetxt('countsresultS',S)

#     a = np.argmax(Z)
#     print('Omega_m :', O[a % len(S)])
#     print('Sigma :', S[a // len(S)])
#         #
#     X, Y = np.meshgrid(O, S)
#     Z = np.exp(Z - Z.max())

#     Z = Z / Z.sum()
#     #     # #
#     t = np.linspace(0, Z.max(), 1000)
#     integral = ((Z >= t[:, None, None]) * Z).sum(axis=(1, 2))
#     # print(integral)
#     f = scipy.interpolate.interp1d(integral, t)
#     t_contours = f(np.array([0.95, 0.68]))

#     ax1 = plt.subplot(121)
#     # ax1.contourf(X, Y, Z)
#     ax1.scatter(O[a % len(S)], S[a // len(S)])
#     ax1.contour(X, Y, Z, t_contours,colors = ['red','blue'],alpha = 0.5)
#     ax1.set_xlabel(r'$\Omega_m$')
#     ax1.set_ylabel(r'$\sigma_8$')

#     ## Plot effective correlation versus data
#     ax2 = plt.subplot(122)
#     #ref
#     xref = r
#     yref = integralXsiMode(xref,cosmo.Cosmology(),mode = R)
#     ax2.plot(xref, yref,color = 'black',linewidth = 1)
#     #data
#     ax2.errorbar(r,xsi.T, yerr=np.sqrt(std.diagonal()),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
#     # refined
#     universe = cosmo.Cosmology(Omega_m=O[a % len(S)],Omega_v=1-O[a % len(S)], sigma_8=S[a // len(S)])
#     # f = lambda k,x : universe.initial_Power_Spectrum_BKKS(k) * np.sin(k*x)/(k*x) * k ** 2 / (2 * np.pi ** 2)
#     # xsi_model = np.array([[intg.quad(lambda k : f(k,x), 0, np.inf,limit=1000)[0] for x in r]])
#     xsi_model = integralXsiMode(xref,universe,mode = R)
#     ax2.scatter(r,xsi[0], color = 'blue',marker = '+')
#     ax2.plot(xref,xsi_model.T, color = 'blue',linestyle = '--',linewidth = 1)

#     # universeRef = cosmo.Cosmology()
#     # xsi_ref = integralXsi(r,universeRef)
#     # ax2.plot(r, xsi_ref,color = 'red',linewidth = 1) #ok equivalent to theory

#     ax2.legend(['Theoretical Convoluted','Refined','Data'])
#     ax2.set_xlabel('Radial distance (Mpc)')
#     ax2.set_ylabel('Correlation function')
#     ax2.set_xlim([15,225])
#     ax2.set_ylim([-0.025,0.20])
#     plt.show()


# # # ############### Refine both R and cosmological parameters together


# # # def refineMaxConvolandCosmo(r,xsi,std,plot = False, mode = 'diagonal'):
# # #     universe = cosmo.Cosmology()
# # #     xsi = np.array([xsi])
# # #     std = np.array(std)
# # #     if mode == 'diagonal':
# # #         Sigma = np.zeros((len(xsi[0]),len(xsi[0])))
# # #         np.fill_diagonal(Sigma,std**2)
# # #         std = Sigma
# # #     invSig = np.linalg.inv(std)

# # #     # print(integralXsi(r,cosmo.Cosmology(Omega_m=0.3,Omega_v=1-0.3, sigma_8=0.8)))
# # #     def loglikelihood(parameters):
# # #         # print(parameters)
# # #         R = parameters[2]
# # #         universe = cosmo.Cosmology(Omega_m=parameters[0],Omega_v=1-parameters[0], sigma_8=parameters[1])
# # #         xsi_model = integralXsiMode(r,universe,mode = R)
# # #         return np.matmul((xsi-xsi_model),np.matmul(invSig,(xsi-xsi_model).T))[0,0]

# # #     sol = opt.minimize(loglikelihood, [0.2,0.7,12],options={ 'disp': True}, bounds = ((0,1),(0,2),(12,14))).x
# # #     # sol = opt.minimize(loglikelihood, [0.2,0.8], bounds = ((0,1),(0,2))).x

# # #     if plot:
# # #         # ref versus data
# # #         ax2 = plt.subplot()
# # #         #ref
# # #         xref = np.linspace(np.min(r),np.max(r),200)
# # #         yref = integralXsiMode(xref,cosmo.Cosmology(),mode = sol[2])
# # #         ax2.plot(xref, yref,color = 'black',linewidth = 1)
# # #         #data
# # #         ax2.errorbar(r,xsi.T, yerr=np.sqrt(std.diagonal()),fmt='none',capsize = 3,ecolor = 'red',elinewidth = 0.7,capthick=0.7)
# # #         # refined
# # #         universe = cosmo.Cosmology(Omega_m=sol[0],Omega_v=1-sol[0], sigma_8=sol[1])
        
# # #         xsi_model = integralXsiMode(xref,universe,sol[2])
# # #         ax2.scatter(r,xsi[0], color = 'blue',marker = '+')
# # #         ax2.plot(xref,xsi_model.T, color = 'blue',linestyle = '--',linewidth = 1)

# # #         ax2.legend(['Theoretical Convoluted','Refined : '+str(list(np.round(sol,3))),'Data'])
# # #         ax2.set_xlabel('Radial distance (Mpc)')
# # #         ax2.set_ylabel('Correlation function')
# # #         ax2.set_xlim([15,225])
# # #         ax2.set_ylim([-0.025,0.20])
# # #         plt.show()

# # #     return(sol) #

# print(refineMaxConvolandCosmo(x,y,std, plot=True))



