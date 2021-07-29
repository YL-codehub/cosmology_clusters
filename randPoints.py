import numpy as np


def randomPoints(norminf,normsup,number,n = 5):
    '''choose random couples of vectors in a 3D box'''
    ijk_1_ijk_2 = np.zeros((1,6))
    for norm in np.linspace(norminf,normsup,number):
        # for rep in range(10):
        n = 5 #number of combinations ?
        ijk = np.random.uniform(size = (n,3))
        invnorms = np.zeros((n,n))
        np.fill_diagonal(invnorms,1/np.linalg.norm(ijk,axis = 1))
        ijk = norm*np.matmul(invnorms, ijk)
        ijk = ijk.astype(int)
        # print(np.linalg.norm(ijk,axis= 1))
        ijk_1 = np.random.randint(0,norm+1,size = (n,3)) #random first vector
        a = (ijk_1>=ijk)
        ijk_2 = ijk_1+np.multiply(1-2*a,ijk) # =  np.multiply(a,ijk_1-ijk)+np.multiply(1-a,ijk_1+ijk)
        temp = np.hstack([ijk_1, ijk_2])
        ijk_1_ijk_2 = np.vstack([ijk_1_ijk_2,temp])    
    return np.unique(ijk_1_ijk_2[1:,:],axis = 0).astype(int) #remove first zero line and identical lines

# # print(randomPoints(1,10,20))
# r = randomPoints(1,100,200)
# # print(np.abs(r[:,0:3]-r[:,3:6]))
# print(np.linalg.norm(r[:,0:3]-r[:,3:6],axis = 1))

def int_sphere(R):
    '''returns int coordinates of points in a sphere of a given radius'''
    radius = np.linspace(1,R,R)
    theta = np.linspace(0,np.pi,10*R,endpoint = False)
    phi = np.linspace(0,2*np.pi,10*R,endpoint = False)
    X = np.array([np.ravel([[[r*np.sin(t)*np.cos(p) for r in radius] for t in theta] for p in phi])]).T
    Y = np.array([np.ravel([[[r*np.sin(t)*np.sin(p) for r in radius] for t in theta] for p in phi])]).T
    Z = np.array([np.ravel([[[r*np.cos(t) for r in radius]  for t in theta] for p in phi])]).T
    res = np.hstack([X,Y,Z])
    res = np.round(res)
    res = res.astype(int)
    res = np.unique(res,axis = 0)
    return res
    
# for el in int_sphere(1):
#     print(el)
# # int_sphere(1)