###########################################################################
# Tests Monte Carlo density (multi-realizations)on correlation likelihood #
###########################################################################
# tester d'abord si refine = max grille
import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt
import scipy.interpolate

fig = plt.figure()
ax = fig.add_subplot(111)

points = np.loadtxt('heavy files/optiBigMC.txt')
sigmas, Oms = points[:,1], points[:,0]

# Show the datapoints on top of this, 
ax.scatter(Oms, sigmas, c='w', s=2, zorder=15, edgecolor='black',alpha=0.75)


# # Use a kernel density estimator to produce local-counts in this space, and grid them to plot.
k = kde.gaussian_kde(points.T)
nbins=200
# xi, yi = np.mgrid[Oms.min():Oms.max():nbins*1j, sigmas.min():sigmas.max():nbins*1j]
xi, yi = np.mgrid[0.1:0.6:nbins*1j, 0.5:1.1:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Show the density
zi = zi/np.sum(zi)
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), zorder=3)
# plt.colorbar()
# zi = zi/np.max(zi)

t = np.linspace(0, zi.max(), 1000)
integral = ((zi >= t[:, None, None]) * zi).sum(axis=(1, 2))
f = scipy.interpolate.interp1d(integral, t)
t_contours = f(np.array([0.95, 0.68]))


# and also the contours. "zorder" sets the vertical order in the plot.
# plt.contour(xi,yi,zi.reshape(xi.shape), zorder=25, colors='0.25')
ax.contour(xi,yi,zi.reshape(xi.shape), t_contours,colors = ['red','blue'],alpha = 0.5)

### Method 2 : 68% polygone plot
###################################
from shapely.geometry.polygon import LinearRing, Polygon
from descartes import PolygonPatch
from scipy.spatial import ConvexHull, convex_hull_plot_2d

center = np.mean(points,axis = 0) #equicentre
distances = np.linalg.norm(points-center,axis = 1)
numberOut95 = int(20-np.round(0.95*20)) #equal to 1, ouf...
numberOut68 = int(20-np.round(0.68*20)) #equal to 6

# print(np.sort(distances))
best68 = np.argsort(distances)[:-numberOut68]
best95 = np.argsort(distances)[:-numberOut95]
# print(points)
# print(best68)
# print(points[best68,:])

best68hull = points[best68,:][ConvexHull(points[best68,:]).vertices,:] #convex hull
best95hull = points[best95,:][ConvexHull(points[best95,:]).vertices,:] #convex hull

# ring68 =  LinearRing([tuple(coord) for coord in best68hull])
# x, y = ring68.xy
# ax.plot(x, y, color='blue', alpha=0.3,linewidth=3, solid_capstyle='round', zorder=2)

ring_mixed95 = Polygon([tuple(coord) for coord in best95hull])
ring_patch95 = PolygonPatch(ring_mixed95,color = 'red')
ax.add_patch(ring_patch95)

ring_mixed68 = Polygon([tuple(coord) for coord in best68hull])
ring_patch68 = PolygonPatch(ring_mixed68,color = 'blue')
ax.add_patch(ring_patch68)

ax.legend([r'$95\%$',r'$68\%$'])

ax.set_ylim(0.5,1.1)
ax.set_xlim(0.1,0.6)
ax.set_xlabel(r'$\Omega_m$')
ax.set_ylabel(r'$\sigma_8$')
plt.show()