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

    def Generate(self, Minf = 5e14, Msup = 1e16, zinf = 0, zsup = 3):
        '''Artificial counts catalog generator, Ms in solar masses ''' # per deg^2
        a,b = m.log(Minf, 10), m.log(Msup, 10)
        self.universe = cosmo.Cosmology(H_0 = H_0, Omega_m = Omega_m, Omega_r = Omega_r, Omega_v = Omega_v, Omega_T = Omega_T, sigma_8 = sigma_8, n_s = n_s)


        Mmin = np.power(10, np.linspace(a, b, int((b - a) / self.logM_intervals) + 1))
        Mmax = Mmin * 10 ** self.logM_intervals
        zmin = np.linspace(zinf, zsup, int((zsup - zinf) / self.z_intervals) + 1)
        zmax = zmin + self.z_intervals

        # means_poisson = self.universe.expected_Counts(Mmin, Mmax, zmin, zmax, mode = 'superarray')
        # N = np.random.poisson(means_poisson)

        nbin = 1

        for i in range(len(Mmin)): #bin [Mm, Mm+Minterval], but exponential dependance
            for j in range(len(zmin)): # bin [zm, zm+z_interval]
                ## Mean counts in the bin thanks to theory
                mean_poisson = self.universe.expected_Counts(Mmin[i],Mmax[i],zmin[j],zmax[j],rad2 = 4*np.pi) # per srad
                N = np.random.poisson(mean_poisson)

                ## Draw a RA and DEC
        #         ## Encode data :
                self.bins[nbin] = {'Mmin':Mmin[i], 'zmin':zmin[j], 'latt':None, 'long':None, 'counts':N}
                # print(self.bins[nbin])
                nbin +=1

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

    def create_catalog_draw(self,delta_new2,dx = 20, number = 60000, plot = True):
        '''from density 3d numpy array (box) to correlated simple catalog for correlation (in a text file). 
    Density field is the probability field and we make a draw in there.
        dx(Mpc) is the physical length of a pixel in the box. 
        The observer is always at nc//2.'''
            
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

        # #select,projection = "mollweide"
        print('Selecting...')
        delta_new2 = Dplus*delta_new2 #must be commented if not Dplus

        print('Drawing points...')
        def pdf():
            '''3D probability probability'''
            b = (delta_new2>=-1)*(1+delta_new2) #method 1
            b = b/np.sum(b)
            return b

        def rejection_method(n,PDf = pdf()):
            '''choose n points with rejection sampling method for a given pdf'''
            M = np.max(PDf)
            N = int(np.round(n*(np.sum(M-PDf)/np.sum(PDf)))*2*6/np.pi) #because many points go in the bin + we points out of the sphere+points twice-drew
            U = np.round((nc-1)*st.uniform().rvs(size=(N,3))).astype(int)
            H = M*st.uniform().rvs(size=N) 
            selection = (PDf[U[:,0],U[:,1],U[:,2]]>=H)
            Uok = U[selection,:]
            sphereTruth = (np.linalg.norm(Uok-nc//2,axis = 1)<=nc//2)
            Uok = Uok[sphereTruth,:]
            indexes = sorted(np.unique(Uok,axis = 0, return_index=True)[1])
            Uok = Uok[indexes,:] # better than just np.unique because np.unique sort values and create bias for the following selection
            return Uok[:np.min([len(Uok),n]),:]

        points = rejection_method(number)

        # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev<np.pi/6)&(elev>0)))
        # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev>np.pi/6)&(elev<np.pi/2)))
        
        Redshifts = r*dx
        selectedRedshifts = np.array([Redshifts[points[:,0],points[:,1],points[:,2]]]).T
        selectedElev = np.array([elev[points[:,0],points[:,1],points[:,2]]]).T
        selectedAz = np.array([az[points[:,0],points[:,1],points[:,2]]]).T
        selected = np.hstack([selectedRedshifts,selectedElev,selectedAz])
        # print(np.mean(selectedRedshifts[selectedElev>0]))
        # print(np.mean(selectedRedshifts[selectedElev>0]))

        print('minLatt : ',np.min(selectedElev*180/np.pi), 'max Latt : ', np.max(selectedElev*180/np.pi))
        print('minLong : ',np.min(selectedAz*180/np.pi), 'max Long : ', np.max(selectedAz*180/np.pi))

        np.savetxt('thresholdcatalog4Comoving.txt',selected)

        print('Number of objects generated : ', selectedRedshifts.shape[0])

        if plot:
            print('Plotting...')
            plt.figure()
            plt.subplot(111, projection="mollweide")
            # plt.subplot(111, projection="polar")
            plt.title("Selected objects")
            plt.grid(True)
            plt.scatter(selectedAz,selectedElev, marker = '.',linewidths=0.01)#, c = selectedRedshifts)
            # plt.scatter(selectedLatt*180/np.pi,selectedRedshifts, marker = '.', c=selectedLong*180/np.pi,linewidths=0.05)
            # cbar = plt.colorbar()
            # cbar.set_label('Redshifts')
            # cbar.set_label('Longitude')
            plt.show()


###########################################################
## Verifications :
#######################cataloger draw###########################

nc = 256
dx = 20
delta_new2 = np.fromfile('boxnc'+str(nc)+'dx'+str(int(dx)))
delta_new2 = np.reshape(delta_new2,(nc,nc,nc))
print(create_catalog_draw(delta_new2,dx = dx, number = 60000))