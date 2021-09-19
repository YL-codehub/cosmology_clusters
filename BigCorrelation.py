from matplotlib import colors
from numpy.linalg.linalg import norm
from scipy.stats.morestats import Mean
import Cosmological_tools as cosmo
import numpy as np
import csv
import matplotlib.pyplot as plt

MultiBins = []
XSIS = []
MeanMass = []
scaleMasses = np.ravel([1.6*10**(14+i*0.1) for i in range(18)])
scaleMasses = scaleMasses*(1+10**0.1)/2 #middle masses

#### Once all correlations are computed (see end of dataCorrelation)

for i in range(40):
    try:
        # Load Corr
        xsis = np.loadtxt('heavy files/CorrBigCorrelation'+str(i)+'.txt')
        XSIS.append(xsis)
         # Load counts 
        temp = np.loadtxt('heavy files/Bins_0.1_0.1BigCorrelationCatalog'+str(i)+'.txt')
        MultiBins.append(np.ravel(temp))
        # Load mean mass
        meanM = np.nan_to_num(np.multiply(np.array(temp) @ np.array([scaleMasses]).T, np.array([1/np.sum(temp,axis = 1)]).T))
        MeanMass.append(np.mean(meanM))
    except:
        continue
    
MultiBins = np.array(MultiBins) # 2D array (ravel(redshiftbins x Mass bins) x catalog)
MultiBins = MultiBins.T
print(MultiBins.shape)
XSIS = np.array(XSIS).T # 2D array (Catalog x bins)
np.savetxt('heavy files/BigCorrXSIs.txt',XSIS)


MultiBins = MultiBins/np.max(np.abs(MultiBins))
XSIS = XSIS/np.max(np.abs(MultiBins))
All = np.vstack([MultiBins,XSIS])
Cov = np.cov(All)
np.savetxt('covBig.txt',Cov)
# (7 Redshifts x 18 Masses)= 126 counts + 21 points de corrélation
corr = np.corrcoef(All)
plt.imshow(corr)
plt.colorbar()
plt.show()

######### Plot 3D of the correlation matrix
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(111, projection='3d')
n1 = 7 * 18
n2 = 21
n = len(corr)
_x = np.arange(n)+0.5
_y = np.arange(n)+0.5
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
top = np.ravel(corr[:n,:n])
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade = True,alpha = 1)
ax1.set_zlim(-1,1)

ax1.set_title('Counts-Correlation correlation matrix')
plt.show()