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

# ##### Adimensionning Counts :
# # Mean mass of a cluster (solar masses)
# MeanMass = np.mean(MeanMass)
# print('Mean mass of a cluster :', MeanMass/1e14, 'x10^14 solar masses')

# # Mean density of the universe (solar masses.Mpc^⁻3)
# rho_m = 135998175800.99973

# # Volume of a bin (Mpc^3) dz = 0.1, et dM = ...
# dz = 418.4544876277076 # Mpc
# scale = np.ravel([1.6*10**(14+i*0.1) for i in range(18)])
# dM = scale*(10**(0.1)-1) # solarMasses
# Vbins = dM*dz # solarMasses.Mpc for each mass bin

# Normalizer = MeanMass/(dM*dz*rho_m)
# Normalizer = np.array([list(Normalizer)*7])
# MultiBins = np.multiply(MultiBins,Normalizer.T)
## Correlation computations
# Xsis = np.loadtxt('heavy files/BigCorrXSIs.txt').T
MultiBins = MultiBins/np.max(np.abs(MultiBins))
XSIS = XSIS/np.max(np.abs(MultiBins))
All = np.vstack([MultiBins,XSIS])
Cov = np.cov(All)
np.savetxt('covBig.txt',Cov)
# (7 Redshifts x 18 Masses)= 126 counts + 21 points de corrélation
# plt.imshow(Cov)
# plt.imshow(Cov)
corr = np.corrcoef(All)
plt.imshow(corr)
plt.colorbar()
plt.show()
