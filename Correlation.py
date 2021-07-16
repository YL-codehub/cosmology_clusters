import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import math as m
import Cosmological_tools as cosmo

def random_Ball(radius, n, Sig = np.identity(3)):
    C = st.multivariate_normal(cov = Sig).rvs(size=n)
    U = st.uniform().rvs(size = n)
    U = radius*U**(1/3)
    S = (U*(C.T/np.linalg.norm(C,axis=1))).T
    return(S)

def plotDraw(N,radius):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    x = radius*np.cos(u) * np.sin(v)
    y = radius*np.sin(u) * np.sin(v)
    z = radius*np.cos(v)

    # alpha controls opacity
    ax.plot_surface(x, y, z, color="g", alpha=0.3)

    #draw random points:
    random = random_Ball(radius,N)
    for coords in random:
        ax.scatter(*coords,marker = 'o')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_title('Points in a sphere')

    plt.show()

# plotDraw(1000,2)

# Landy and Szaslay estimator for correlation data

def cart2sph(x,y,z):
    '''from 3D cartesian coordinates to spherical ones'''
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # lambda
    az = m.atan2(y,x)                           # phi
    return r, elev, az # rad

def cart2sphA(pts):
    '''list of 3D cartesian coordinates to list of spherical geographic ones'''
    return np.array([cart2sph(x,y,z) for x,y,z in pts])

def distanceAB_spherical(A,B):
    return(np.sqrt(abs(A[0]**2+B[0]**2-2*A[0]*B[0]*(np.cos(A[1])*np.cos(B[1])*np.cos(A[2]-B[2])+np.sin(A[1])*np.sin(B[1])))))

def number_Pairs(Data1, Data2, rmin, rmax):
    '''number of pairs Data1(list of spherical coordinates) x Data2 (idem) with a mutual distance betw rmin (strictly) and rmax'''
    counts = 0
    for elt1 in Data1:
        for elt2 in Data2:
            r = distanceAB_spherical(elt1,elt2)
            if r >= rmin and r <= rmax:
                counts+=1
    return(counts)

def CorrelationLS(r,dr,Data, rData):
    '''correlation in bin r-dr/2,r+dr/2 in Data (list of spherical coordinates), with the comparison to rData random mapping.
    Landy and Szaslay estimator formula is used.'''
    N = len(Data)
    rmin, rmax = r-dr/2, r+dr/2
    DD = number_Pairs(Data,Data,rmin,rmax)
    RD = number_Pairs(Data,rData,rmin,rmax)
    RR = number_Pairs(rData,rData,rmin,rmax)
    print(DD,RD,RR)
    return((DD-2*RD+RR)/RR)

#
rData1 = random_Ball(10,500)
rData2 = random_Ball(10,500)
# # print(CorrelationLS(5,0.01,rData1,rData2))
#
x = np.linspace(1,10, 19)
y = [CorrelationLS(el,0.01,rData1,rData2) for el in x]
plt.plot(x,y)
plt.show()