from numpy import array
from scipy.interpolate import RegularGridInterpolator as rgi
import numpy as np

nc = 256
dx = 20
delta_new = np.fromfile('heavy files/box0nc'+str(nc)+'dx'+str(int(dx)))
delta_new = np.reshape(delta_new,(nc,nc,nc))

def pdf():
    '''3D probability probability'''
    h = (delta_new>=-1)*(1+delta_new)
    h = h/np.sum(h)
    return h

pdf_box = pdf()
x = range(256)
my_interpolating_function = rgi((x,x,x), pdf_box)
# print(pdf_box[1,1,1], my_interpolating_function([[1,1,1]]))


    # def rejection_method(n,PDf = pdf()):
    #     '''choose n points with rejection sampling method for a given pdf'''
    #     M = np.max(PDf)
    #     N = int(np.round(2*n*(np.sum(M-PDf)/np.sum(PDf)))*2*6/np.pi) #because many points go in the bin + we points out of the sphere+points twice-drew
    #     U = np.round((nc-1)*st.uniform().rvs(size=(N,3))).astype(int)
    #     H = M*st.uniform().rvs(size=N) 
    #     selection = (PDf[U[:,0],U[:,1],U[:,2]]>=H)
    #     Uok = U[selection,:]
    #     sphereTruth = (np.linalg.norm(Uok-nc//2,axis = 1)<=nc//2)
    #     Uok = Uok[sphereTruth,:]
    #     indexes = sorted(np.unique(Uok,axis = 0, return_index=True)[1])
    #     Uok = Uok[indexes,:] # better than just np.unique because np.unique sort values and create bias for the following selection
    #     return Uok[:np.min([len(Uok),n]),:]

    # points = rejection_method(number)