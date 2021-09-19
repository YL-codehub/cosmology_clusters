from numpy.core.fromnumeric import mean
import scipy.optimize as opt
import Cosmological_tools as cosmo
import numpy as np
import matplotlib.pyplot as plt
import math as m 
import csv
import time as t
import scipy.interpolate
import scipy.stats as st


### Corentin's functions (verified)
def convert_cartesian_to_sky_full_angle(X,Y,Z):
    RA = np.arctan2(X,Z)
    DEC = np.arcsin(Y/np.sqrt(X**2 + Y**2 + Z**2))
    r = np.sqrt(X**2 + Y**2 + Z**2)
    return(r,RA,DEC) #RA = phi, DEC = pi/2-theta

def convert_sky_to_cartesian_full_angle(r,RA,DEC):
    X = r*np.sin(RA)*np.cos(DEC)
    Y = r*np.sin(DEC)
    Z = r*np.cos(RA)*np.cos(DEC)
    return(X,Y,Z)


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

def radialcomov_to_redshift(r):
    '''Gives Redshift associated to a given comoving radial distance'''
    Cosmo = cosmo.Cosmology() # usual Omega by default
    func = lambda z : Cosmo.comoving_Distance(z)-r
    dfunc = lambda z : Cosmo.c/Cosmo.H(z,mode = 'z')
    z = opt.newton(func=func,x0=0,fprime=dfunc)
    return(z)

def cart2sph(x,y,z,mode = 'default'):
    '''from 3D cartesian coordinates to spherical ones'''
    if mode == 'default':
        XsqPlusYsq = x**2 + y**2
        r = m.sqrt(XsqPlusYsq + z**2)               # r
        elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # lambda
        az = m.atan2(y,x)                           # phi
        return r, elev, az # rad
    elif mode == 'np':
        XsqPlusYsq = np.power(x,2) + np.power(y,2)
        r = np.sqrt(XsqPlusYsq + np.power(z,2))               # r
        elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # lambda
        az = np.arctan2(y,x)                           # phi
        return r, elev, az # rad

class doubleCatalog:
    def __init__(self, inputfile = None, logM_intervals = 0.1, z_intervals = 0.1):

        self.file = inputfile
        self.entries = []                     # [ {'M':,'z':,'latt':, 'long':}, ...]
        self.logM_intervals = logM_intervals  # for artificial data use only
        self.z_intervals = z_intervals        # for artificial data use only
        self.bins = {}                       # [ {'Mmin':,'zmin':,'latt':, 'long':, 'counts':}, ...] , a bin is [Mmin,Mmin+M_intervals] x [zmin, zmin+z_intervals]
        self.universe = None
        self.Mmin = None
        self.Mmax = None
        self.zmin = None
        self.zmax = None
        self.counts = None # z x M matrix
        self.nc = 256

    def Generate(self, Minf = 1.6e14, Msup = 1e16, zinf = 0, zsup = 3, name = 'Catalog'):
        '''Artificial counts catalog generator, Ms in solar masses ''' # per deg^2
        zsup = radialcomov_to_redshift((self.nc/2-1)*dx)-0.1 #avoid border effects
        print(zsup)
        a,b = m.log(Minf, 10), m.log(Msup, 10)
        self.universe = cosmo.Cosmology()#H_0 = H_0, Omega_m = Omega_m, Omega_r = Omega_r, Omega_v = Omega_v, Omega_T = Omega_T, sigma_8 = sigma_8, n_s = n_s)


        self.Mmin = np.power(10, np.linspace(a, b, int((b - a) / self.logM_intervals) + 1))
        self.Mmax = self.Mmin * 10 ** self.logM_intervals
        self.zmin = np.linspace(zinf, zsup, int((zsup - zinf) / self.z_intervals) + 1)
        self.zmax = self.zmin + self.z_intervals

        # means_poisson = self.universe.expected_Counts(Mmin, Mmax, zmin, zmax, mode = 'superarray')
        # N = np.random.poisson(means_poisson)

        nbin = 1
        self.counts = np.zeros((len(self.zmin),len(self.Mmin)))
        for i in range(len(self.Mmin)): #bin [Mm, Mm+Minterval], but exponential dependance
            for j in range(len(self.zmin)): # bin [zm, zm+z_interval]
                ## Mean counts in the bin thanks to theory
                mean_poisson = self.universe.expected_Counts(self.Mmin[i],self.Mmax[i],self.zmin[j],self.zmax[j],rad2 = 4*np.pi) # per srad
                N = np.random.poisson(mean_poisson)

                ## Draw a RA and DEC
        #         ## Encode data :
                self.bins[nbin] = {'Mmin':self.Mmin[i], 'zmin':self.zmin[j], 'latt':None, 'long':None, 'counts':N}
                self.counts[j,i] = N
                # print(self.bins[nbin])
                nbin +=1
        np.savetxt('heavy files/Bins_'+str(self.z_intervals)+'_'+str(self.logM_intervals)+name+'.txt',self.counts)
        # for i in range(len(Mmin)): #bin [Mm, Mm+Minterval], but exponential dependance
        #     for j in range(len(zmin)): # bin [zm, zm+z_interval]
        # #         ## Encode data :
        #         self.bins[nbin] = {'Mmin':Mmin[i], 'zmin':zmin[j], 'latt':None, 'long':None, 'counts':N[i,j]}
        #         # print(self.bins[nbin])
        #         nbin +=1

    def GenerateCorrelatedCounts(self,delta_new2, Minf = 1.6e14, Msup = 1e16, zinf = 0, zsup = 3, name = 'Catalog'):
        # To be integrated in double draw cataloger...
        '''Artificial counts catalog generator, Ms in solar masses ''' # per deg^2
        zsup = radialcomov_to_redshift((self.nc/2-1)*dx)-0.1 #avoid border effects
        print(zsup)
        a,b = m.log(Minf, 10), m.log(Msup, 10)
        self.universe = cosmo.Cosmology()#H_0 = H_0, Omega_m = Omega_m, Omega_r = Omega_r, Omega_v = Omega_v, Omega_T = Omega_T, sigma_8 = sigma_8, n_s = n_s)


        self.Mmin = np.power(10, np.linspace(a, b, int((b - a) / self.logM_intervals) + 1))
        self.Mmax = self.Mmin * 10 ** self.logM_intervals
        self.zmin = np.linspace(zinf, zsup, int((zsup - zinf) / self.z_intervals) + 1)
        self.zmax = self.zmin + self.z_intervals

        # means_poisson = self.universe.expected_Counts(Mmin, Mmax, zmin, zmax, mode = 'superarray')
        # N = np.random.poisson(means_poisson)
        nc = len(delta_new2)
        
        print('Arraying...')
        b = np.array([[[[i,j,k] for i in range(nc)] for j in range(nc)] for k in range(nc)])
        print('Sphericalizing...')
        mid = len(b)//2
        r, elev, az = cart2sph(b[:,:,:,0]-mid,b[:,:,:,1]-mid,b[:,:,:,2]-mid,mode = 'np')

        # # No Newton
        print('Redshifting...')
        from scipy import interpolate
        step = 0.01
        zmax = radialcomov_to_redshift(r.max()*dx)+0.1
        Z = np.linspace(0,zmax, int((zmax-0)/step)+1)
        temp = cosmo.Cosmology()
        R = [temp.comoving_Distance(z) for z in Z]
        f = interpolate.interp1d(R, Z)
        Redshifts = f(r*dx)


        Dp = [temp.Dplus(z,mode = 'np') for z in Z]
        g = interpolate.interp1d(Z,Dp)
        Dplus = g(Redshifts)

        def Bias_MoWhite(z,M):
            '''returns Mo & WHite bias for a given z  array and M a solar Mass'''
            delta_c = 1.686
            D = g(z)
            delta_1 = delta_c/D
            sig0 = self.universe.initial_sigma(M*1e15,mode = 'quad')
            nu_1 = delta_1/sig0
            Bias = 1+(np.power(nu_1,2)-1)/delta_1
            return(Bias)

        nbin = 1
        self.counts = np.zeros((len(self.zmin),len(self.Mmin)))
        for i in range(len(self.Mmin)): #bin [Mm, Mm+Minterval], but exponential dependance
            for j in range(len(self.zmin)): # bin [zm, zm+z_interval]
                ## Mean counts in the bin thanks to theory
                midz = (self.zmin[j]+self.zmax[j])/2
                midM = (self.Mmin[i]+self.Mmax[i])/2
                mean_poisson = self.universe.expected_Counts(self.Mmin[i],self.Mmax[i],self.zmin[j],self.zmax[j],rad2 = 4*np.pi) # per srad
                delta = np.mean(delta_new2[(self.zmin[j]<=Redshifts)*(Redshifts<=self.zmax[j])])
                New_mean_poisson = mean_poisson*(1+g(midz)*Bias_MoWhite(midz,midM/1e15)*delta)
                N = np.random.poisson(New_mean_poisson)
                
                ## Draw a RA and DEC
        #         ## Encode data :
                self.bins[nbin] = {'Mmin':self.Mmin[i], 'zmin':self.zmin[j], 'latt':None, 'long':None, 'counts':N}
                self.counts[j,i] = N
                # print(self.bins[nbin])
                nbin +=1
        np.savetxt('heavy files/CorrelatedBins_'+str(self.z_intervals)+'_'+str(self.logM_intervals)+name+'.txt',self.counts)
        # for i in range(len(Mmin)): #bin [Mm, Mm+Minterval], but exponential dependance
        #     for j in range(len(zmin)): # bin [zm, zm+z_interval]
        # #         ## Encode data :
        #         self.bins[nbin] = {'Mmin':Mmin[i], 'zmin':zmin[j], 'latt':None, 'long':None, 'counts':N[i,j]}
        #         # print(self.bins[nbin])
        #         nbin +=1


    def save_bins(self,name):
        a_file = open(name, "wb")
        pk.dump(self.bins, a_file)
        a_file.close()

    def read_bins(self,name):
        a_file = open(name, "rb")
        self.bins = pk.load(a_file)

    def create_catalog_draw(self,delta_new2,dx = 20, plot = True, name = 'Catalog', Growthfactor = False, MoWhite = False):
        '''from density 3d numpy array (box) to correlated simple catalog for correlation (in a text file). 
    Density field is the probability field and we make a draw in there.
        dx(Mpc) is the physical length of a pixel in the box. 
        The observer is always at nc//2.'''        # #select,projection = "mollweide"
        
        print('Arraying...')
        b = np.array([[[[i,j,k] for i in range(nc)] for j in range(nc)] for k in range(nc)])
        print('Sphericalizing...')
        mid = len(b)//2
        r, elev, az = cart2sph(b[:,:,:,0]-mid,b[:,:,:,1]-mid,b[:,:,:,2]-mid,mode = 'np')

        # # No Newton
        print('Redshifting...')
        from scipy import interpolate
        step = 0.01
        zmax = radialcomov_to_redshift(r.max()*dx)+0.1
        Z = np.linspace(0,zmax, int((zmax-0)/step)+1)
        temp = cosmo.Cosmology()
        R = [temp.comoving_Distance(z) for z in Z]
        f = interpolate.interp1d(R, Z)
        Redshifts = f(r*dx)


        Dp = [temp.Dplus(z,mode = 'np') for z in Z]
        g = interpolate.interp1d(Z,Dp)
        Dplus = g(Redshifts)

        print('Selecting...')
        if Growthfactor:
            delta_new2 = Dplus*delta_new2 #must be commented if not Dplus

        def Bias_MoWhite(z,M):
            '''returns Mo & WHite bias for a given z  array and M a solar Mass'''
            delta_c = 1.686
            D = g(z)
            delta_1 = delta_c/D
            sig0 = self.universe.initial_sigma(M*1e15,mode = 'quad')
            nu_1 = delta_1/sig0
            Bias = 1+(np.power(nu_1,2)-1)/delta_1
            return(Bias)

        ###############################
        ## Testing Mo and White bias ##
        ###############################
        # xref, yref = readtxt('Vérifications fonctions/biasz.txt')
        # xref,yref = np.ravel(xref),np.ravel(yref)
        # xref,yref = xref[xref<zmax],yref[xref<zmax]
        # plt.plot(xref, yref,color = 'black',linewidth = 1)
        # y = Bias_MoWhite(xref,0.1)
        # plt.plot(xref,y, color = 'red',linestyle = '--',linewidth = 1)
        # plt.legend(['Reference','Mine'])
        # plt.xlabel('Redshift')
        # plt.ylabel('Mo and White bias at M = 0.1 solar Masses')
        # plt.show()


        print('Drawing points...')
        def Shellpdf(z,dz = self.z_intervals,mean_M = 1e14):
            '''3D probability probability'''
            # Bias = Bias_MoWhite(Redshifts,np.ravel([mean_M]))
            Bias = 1
            if MoWhite:
                Bias = Bias_MoWhite(Redshifts,mean_M)
            print('Max number of objects in the shell : ', np.sum((z - dz / 2 <= Redshifts) * (Redshifts <= z + dz / 2)))
            b = (z-dz/2<=Redshifts)*(Redshifts<=z+dz/2)*(delta_new2>=-1)*(1+Bias*delta_new2) #method 1
            b = b/np.sum(b)
            return b

        def rejection_method(n,PDf):
            '''choose n points with rejection sampling method for a given pdf'''
            M = np.max(PDf)
            # print(np.sum(M-PDf)/np.sum(PDf))
            N = int(np.nan_to_num(np.round(n*(np.sum(M-PDf)/np.sum(PDf)))*2*6/np.pi)) #because many points go in the bin + we points out of the sphere+points twice-drew
            N = np.min([N,7000000]) #un peu sombre
            # N = 500000
            print('Number of draw:',N)
            U = np.round((nc-1)*st.uniform().rvs(size=(N,3))).astype(int)
            H = M*st.uniform().rvs(size=N) 
            selection = (PDf[U[:,0],U[:,1],U[:,2]]>=H)
            Uok = U[selection,:]
            sphereTruth = (np.linalg.norm(Uok-nc//2,axis = 1)<=nc//2)
            Uok = Uok[sphereTruth,:]
            indexes = sorted(np.unique(Uok,axis = 0, return_index=True)[1])
            Uok = Uok[indexes,:] # better than just np.unique because np.unique sort values and create bias for the following selection
            print('Number of unique objects drawn :', len(Uok))
            return Uok[:np.min([len(Uok),n]),:]

        points = np.zeros((1,3),dtype=int)
        masses = np.zeros((1,1),dtype=int)
        Ms = (self.Mmin+self.Mmax)/2
        for i in range(len(self.zmin)):
            print('-----------------------------------------')
            # Draw points 
            Mdistrib = self.counts[i]
            print("Central Shell Redshift: ",(self.zmin[i]+self.zmax[i])/2)
            n = int(np.sum(Mdistrib))
            print("objects to draw in the shell: ",n)
            meanM = np.nan_to_num(np.sum(Mdistrib*Ms)/n)
            newpoints = rejection_method(n,PDf = Shellpdf((self.zmin[i]+self.zmax[i])/2,mean_M = meanM))
            print("effective number of draw: ",newpoints.shape[0])
            points = np.vstack([points,newpoints])
            #Draw masses (the mire density = the bigger the affected mass):
            tempMasses = np.array([np.hstack([[Ms[i]]*int(Mdistrib[i]) for i in range(len(Mdistrib))])]).T
            indexes = np.argsort(delta_new2[newpoints[:,0],newpoints[:,1],newpoints[:,2]])
            Rev_indexes = np.argsort(indexes)
            newMasses = tempMasses[Rev_indexes]
            masses = np.vstack([masses,newMasses])
        print('-----------------------------------------')
        points = points[1:,:]
        SelectedMasses = masses[1:,:]

        ### Create the associated random catalog

        random_points = np.zeros((1,3),dtype=int)

        dz = self.z_intervals
        for i in range(len(self.zmin)):
            z = (self.zmin[i]+self.zmax[i])/2
            Mdistrib = self.counts[i]
            n = 2*int(np.sum(Mdistrib)) # 10 times bigger random catalog
            Nr = 600*n
            print(Nr)
            Nr = min(Nr, 10000000)
            # Nr = min(Nr,500000)
            U = np.round((self.nc-1)*st.uniform().rvs(size=(Nr,3))).astype(int)
            redshifts = Redshifts[U[:,0],U[:,1],U[:,2]]
            new_random_points = U[(z-dz/2<=redshifts)*(redshifts<=z+dz/2)]
            
            indexes = sorted(np.unique(new_random_points,axis = 0, return_index=True)[1])
            new_random_points = new_random_points[indexes,:] # better than just np.unique because np.unique sort values and create bias for the following selection
            new_random_points =  new_random_points[:np.min([len(new_random_points),n]),:]
            print("Random Catalog shell ok: ",len(new_random_points)==n)
            random_points = np.vstack([random_points,new_random_points])
        print('-----------------------------------------')
        random_points = random_points[1:,:]

        # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev<np.pi/6)&(elev>0)))
        # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev>np.pi/6)&(elev<np.pi/2)))
        
        Redshifts = r*dx # go back to comoving coordinates 
        selectedRedshifts = np.array([Redshifts[points[:,0],points[:,1],points[:,2]]]).T
        selectedElev = np.array([elev[points[:,0],points[:,1],points[:,2]]]).T
        selectedAz = np.array([az[points[:,0],points[:,1],points[:,2]]]).T
        selected = np.hstack([selectedRedshifts,selectedElev,selectedAz,SelectedMasses])
        # print(np.mean(selectedRedshifts[selectedElev>0]))
        # print(np.mean(selectedRedshifts[selectedElev>0]))

        print('minLatt : ',np.min(selectedElev*180/np.pi), 'max Latt : ', np.max(selectedElev*180/np.pi))
        print('minLong : ',np.min(selectedAz*180/np.pi), 'max Long : ', np.max(selectedAz*180/np.pi))

        np.savetxt('heavy files/'+name+'.txt',selected)

        print('Number of objects generated : ', selectedRedshifts.shape[0])

        ## Save random Catalog
        RselectedRedshifts = np.array([Redshifts[random_points[:,0],random_points[:,1],random_points[:,2]]]).T
        RselectedElev = np.array([elev[random_points[:,0],random_points[:,1],random_points[:,2]]]).T
        RselectedAz = np.array([az[random_points[:,0],random_points[:,1],random_points[:,2]]]).T
        Rselected = np.hstack([RselectedRedshifts,RselectedElev,RselectedAz])

        np.savetxt('heavy files/RANDOM'+name+'.txt',Rselected)

        print('Number of random objects generated : ', RselectedRedshifts.shape[0]) 

        if plot:
            print('Plotting...')
            plt.figure()
            plt.subplot(111, projection="mollweide")
            # plt.subplot(111, projection="polar")
            plt.title("Selected objects")
            plt.grid(True)
            plt.scatter(selectedAz,selectedElev, marker = '.',linewidths=0.01,color = 'blue')#, c = selectedRedshifts)
            plt.scatter(RselectedAz,RselectedElev, marker = '.',linewidths=0.01,color = 'red',alpha = 0.1)
            # plt.scatter(selectedLatt*180/np.pi,selectedRedshifts, marker = '.', c=selectedLong*180/np.pi,linewidths=0.05)
            # cbar = plt.colorbar()
            # cbar.set_label('Redshifts')
            # cbar.set_label('Longitude')
            plt.show()


###########################################################
## Verifications :
# #######################cataloger draw###########################

nc = 256
dx = 20
# # print(create_catalog_draw(delta_new2,dx = dx, number = 60000))
# temp = doubleCatalog()
# temp.GenerateCorrelatedCounts(delta_new2)
# print(temp.entries)
# #######################Mo & White Bias###########################
for i in range(20):
    print('Iteration '+str(i))
    delta_new2 = np.fromfile("heavy files/"+'boxnc'+str(nc)+'dx'+str(int(dx)))
    delta_new2 = np.reshape(delta_new2,(nc,nc,nc))
    temp = doubleCatalog()
    # temp.Generate(name = 'BigCorrelationCatalog'+str(i))
    temp.Generate(name = 'BigDoubleCatalog'+str(i))
    temp.create_catalog_draw(delta_new2,plot = False, name = 'BigDoubleCatalog'+str(i), Growthfactor = False, MoWhite = False)
    print('------------')

# res = temp.Bias_MoWhite(np.linspace(0.1,1,11),np.linspace(1e14,1e16,12))

# # # # #ref
# # # # xref, yref = readtxt('Vérifications fonctions/biasM.txt')
# # # # plt.plot(xref, yref,color = 'black',linewidth = 1)
# # # # y = temp.Bias_MoWhite(np.ravel([1]),np.ravel(xref))[0]
# # # # print(y)
# # # # plt.plot(xref,y, color = 'red',linestyle = '--',linewidth = 1)
# # # # plt.legend(['Reference','Mine'])
# # # # plt.xlabel('Mass (1e15 solar masses)')
# # # # plt.ylabel('Mo and White bias at z = 1')
# # # # plt.show()
#

# def bins_from_catalog(filename,z_interval,logM_intervals):
#     Catalog = np.loadtxt('filename')


