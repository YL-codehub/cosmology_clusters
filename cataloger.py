import scipy.optimize as opt
import Cosmological_tools as cosmo
import numpy as np
import matplotlib.pyplot as plt
import math as m 
import csv
import time as t
from scipy.interpolate import RegularGridInterpolator as rgi
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

def create_catalog_threshold(delta_new2,dx = 20, threshold = 1.686, plot = True):
    '''from density 3d numpy array (box) to correlated simple catalog for correlation (in a text file). 
    Any overdensity contrast over treshold is considered to be a cluster.
    dx(Mpc) is the physical length of a pixel in the box. 
    The observer is always at nc//2.'''
        
    nc = len(delta_new2)

    print('Arraying...')
    b = np.array([[[[i,j,k] for k in range(nc)] for j in range(nc)] for i in range(nc)])
    print('Sphericalizing...')
    mid = len(b)//2
    r, az, elev = convert_cartesian_to_sky_full_angle(b[:,:,:,0]-mid,b[:,:,:,1]-mid,b[:,:,:,2]-mid)
    
   
    # # No Newton
    # print('Redshifting...')
    # from scipy import interpolate
    # step = 0.01
    # zmax = radialcomov_to_redshift(r.max()*dx)+0.1
    # Z = np.linspace(0,zmax, int((zmax-0)/step)+1)
    # temp = cosmo.Cosmology()
    # R = [temp.comoving_Distance(z) for z in Z]
    # f = interpolate.interp1d(R, Z)
    # Redshifts = f(r*dx)


    # Dp = [temp.Dplus(z,mode = 'np') for z in Z]
    # g = interpolate.interp1d(Z,Dp)
    # Dplus = g(Redshifts)

    # #select,projection = "mollweide"
    print('Selecting...')
    # delta_new2 = Dplus*delta_new2 #must be commented if not Dplus? no i don't think so
    Redshifts = r*dx #must be commented if Dplus #must be commented if Dplus
    selection = (delta_new2>threshold)
    # zmaxsphere = radialcomov_to_redshift(nc//2*dx)
    zmaxsphere = nc//2*dx  #must be commented if Dplus
    print('maxsphere',zmaxsphere)
    selection = (delta_new2>threshold)&(Redshifts<=zmaxsphere)

    # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev<np.pi/6)&(elev>0)))
    # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev>np.pi/6)&(elev<np.pi/2)))
    
    selectedRedshifts = np.array([Redshifts[selection]])
    selectedElev = np.array([elev[selection]])
    selectedAz = np.array([az[selection]])
    selected = np.hstack([selectedRedshifts.T,selectedElev.T,selectedAz.T])
    # print(np.mean(selectedRedshifts[selectedElev>0]))
    # print(np.mean(selectedRedshifts[selectedElev>0]))

    print('minLatt : ',np.min(selectedElev*180/np.pi), 'max Latt : ', np.max(selectedElev*180/np.pi))
    print('minLong : ',np.min(selectedAz*180/np.pi), 'max Long : ', np.max(selectedAz*180/np.pi))

    np.savetxt('thresholdcatalog.txt',selected)

    print('Number of objects generated : ', selectedRedshifts.shape[1])

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

def create_catalog_draw(delta_new2,dx = 20, number = 10000, plot = True, mode ='discrete', MonteCarloIndex = 0):
    '''from density 3d numpy array (box) to correlated simple catalog for correlation (in a text file). 
   Density field is the probability field and we make a draw in there.
    dx(Mpc) is the physical length of a pixel in the box. 
    The observer is always at nc//2.'''
        
    nc = len(delta_new2)
    mid = nc//2

    if mode == 'discrete':
    ## Discrete Method
        print('Arraying...')
        b = np.array([[[[i,j,k] for i in range(nc)] for j in range(nc)] for k in range(nc)])
        print('Sphericalizing...')
        r, az, elev = convert_cartesian_to_sky_full_angle(b[:,:,:,0]-mid,b[:,:,:,1]-mid,b[:,:,:,2]-mid)

    # # No Newton
    # print('Redshifting...')
    # from scipy import interpolate
    # step = 0.01
    # zmax = radialcomov_to_redshift(r.max()*dx)+0.1
    # Z = np.linspace(0,zmax, int((zmax-0)/step)+1)
    # temp = cosmo.Cosmology()
    # R = [temp.comoving_Distance(z) for z in Z]
    # f = interpolate.interp1d(R, Z)
    # Redshifts = f(r*dx)


    # Dp = [temp.Dplus(z,mode = 'np') for z in Z]
    # g = interpolate.interp1d(Z,Dp)
    # Dplus = g(Redshifts)

    # #select,projection = "mollweide"
    # print('Selecting...')
    # delta_new2 = Dplus*delta_new2 #must be commented if not Dplus

    print('Drawing points...')
    # c = -1/np.min(delta_new2)
    # c = 0.2 #valeur commune
    def pdf():
        '''3D probability probability'''
        h = (delta_new2>=-1)*(1+delta_new2) #method 1
        # h = delta_new2-np.min(delta_new2) #method 2
        # h = 1+c*delta_new2
        h = h/np.sum(h)
        return h

    ##### Discrete Rejection Method 
    if mode == 'discrete':
        def rejection_method(n,PDf = pdf()):
            '''choose n points with rejection sampling method for a given pdf'''
            M = np.max(PDf)
            N = int(np.round(2*n*(np.sum(M-PDf)/np.sum(PDf)))*2*6/np.pi) #because many points go in the bin + we points out of the sphere+points twice-drew
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

    elif mode == 'continuous':
    ##### Continous Rejection Method 
        pdf_box = pdf()
        x = range(nc)
        pdf_func = rgi((x,x,x), pdf_box)

        def Continuous_rejection_method(n,PDf = pdf_func):
            '''choose n points with rejection sampling method for a given pdf'''
            M = np.max(pdf_box)
            N = int(np.round(2*n*(np.sum(M-pdf_box)/np.sum(pdf_box)))*2*6/np.pi) #because many points go in the bin + we points out of the sphere+points twice-drew
            U = np.random.uniform(0,nc-1,size = (N,3))
            H = M*st.uniform().rvs(size=N) 
            print('step 1')
            selection = (PDf(U)>=H)
            print('step 2')
            Uok = U[selection,:]
            sphereTruth = (np.linalg.norm(Uok-nc//2,axis = 1)<=nc//2)
            Uok = Uok[sphereTruth,:]
            # indexes = sorted(np.unique(Uok,axis = 0, return_index=True)[1])
            # Uok = Uok[indexes,:] # better than just np.unique because np.unique sort values and create bias for the following selection
            return Uok[:np.min([len(Uok),n]),:]
        
        points = Continuous_rejection_method(number)
    

    # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev<np.pi/6)&(elev>0)))
    # print(np.sum((delta_new2>threshold)&(Redshifts<=zmaxsphere)&(elev>np.pi/6)&(elev<np.pi/2)))
    
    if mode == 'discrete':
    ## Discrete selection :
        Redshifts = r*dx
        selectedRedshifts = np.array([Redshifts[points[:,0],points[:,1],points[:,2]]]).T
        selectedElev = np.array([elev[points[:,0],points[:,1],points[:,2]]]).T
        selectedAz = np.array([az[points[:,0],points[:,1],points[:,2]]]).T
        selected = np.hstack([selectedRedshifts,selectedElev,selectedAz])
        # # # print(np.mean(selectedRedshifts[selectedElev>0]))
        # # # print(np.mean(selectedRedshifts[selectedElev>0]))
        np.savetxt('heavy files/BigCatalogMC'+str(MonteCarloIndex)+'.txt',selected)

    elif mode == 'continuous':
    # Continuous selection :
        points = points -mid #centering
        points *= dx # going to Mpc
        selectedRedshifts, selectedAz, selectedElev = convert_cartesian_to_sky_full_angle(points[:,0],points[:,1],points[:,2])
        ## selectedRedshifts = comoving heres
        selected = np.hstack([np.array([selectedRedshifts]).T,np.array([selectedElev]).T,np.array([selectedAz]).T])
        np.savetxt('heavy files/BigCatalogMCContinuous'+str(MonteCarloIndex)+'.txt',selected)


    print('minLatt : ',np.min(selectedElev*180/np.pi), 'max Latt : ', np.max(selectedElev*180/np.pi))
    print('minLong : ',np.min(selectedAz*180/np.pi), 'max Long : ', np.max(selectedAz*180/np.pi))

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
#######################redshift converter##################

# temp  = cosmo.Cosmology()
# x = np.linspace(0,4,100)
# y = [temp.comoving_Distance(z) for z in x]
# plt.plot(x,y)
# plt.xlabel('Redshift')
# plt.ylabel('Comoving distance (Mpc)')

# r0 = 7000
# z0 = radialcomov_to_redshift(r0)
# plt.scatter(z0,r0,marker = '+',color = 'red')
# plt.show()

#######################cataloger threshold###########################


# nc = 256
# dx = 20
# delta_new2 = np.fromfile('boxnc'+str(nc)+'dx'+str(int(dx)))
# delta_new2 = np.reshape(delta_new2,(nc,nc,nc))
# print(np.max(delta_new2),np.std(delta_new2))
# create_catalog_threshold(delta_new2,threshold = 2.15,dx = dx)


# print(radialcomov_to_redshift(20*512))

# ET SI LA DENSITE A UN CONSTRASTE < -1 c'est pas normal
# print('min density: ',np.min(delta_new2))

#######################cataloger draw###########################

# nc = 256
# dx = 20
# delta_new2 = np.fromfile('heavy files/box2nc'+str(nc)+'dx'+str(int(dx)))
# delta_new2 = np.reshape(delta_new2,(nc,nc,nc))
# print(np.min(delta_new2),np.std(delta_new2))
# create_catalog_draw(delta_new2,dx = dx, number = 10000,MonteCarloIndex=0,plot = True)

#######################cataloger MonteCarlo###########################

# nc = 256
# dx = 20
# delta_new2 = np.fromfile('heavy files/boxnc'+str(nc)+'dx'+str(int(dx)))
# delta_new2 = np.reshape(delta_new2,(nc,nc,nc))
# # print(create_catalog_draw(delta_new2,dx = dx, number = 80000))

# for i in range(20):
#     print('Iteration: ',i)
#     print('-------------------')
#     create_catalog_draw(delta_new2,dx = dx, number = 500000,plot = False,MonteCarloIndex=i)

#######################cataloger MonteCarlo on different bx###########################

nc = 256
dx = 20
# print(create_catalog_draw(delta_new2,dx = dx, number = 80000))

for i in range(20):
    print('Iteration: ',i)
    print('-------------------')
    delta_new2 = np.fromfile('heavy files/box'+str(i)+'nc'+str(nc)+'dx'+str(int(dx)))
    delta_new2 = np.reshape(delta_new2,(nc,nc,nc))  
    create_catalog_draw(delta_new2,dx = dx, number = 200000,plot = False, MonteCarloIndex=i,mode = 'continuous')



