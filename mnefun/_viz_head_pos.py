# -*- coding: utf-8 -*-

"""script to viz individual cHPI data

# Authors : Kambiz Tavabi <ktavabi@gmail.com>
            Maggie Clarke
"""

import numpy as np
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from mne.chpi import (read_head_pos, head_pos_to_trans_rot_t)


def _apply_affine(loc, rot, trans):
    """return affine transformed coordinates for input Cartesian coords

    Parameters
    ----------
    loc : array
        (3,) location to be transformed
    rot : array
        (3, 3) rotation matrix
    trans : array
        (3,) translation vector
    """
    return rot.dot(loc) + trans


np.set_printoptions(precision=8, suppress=True)

# Get cHPI data
pos = read_head_pos('/Users/ktavabi/Projects/Sandbox/bad_101_am_raw.pos')
translations, rotations, times = head_pos_to_trans_rot_t(pos)

init_pos = translations[0]  # take first head position (1st trans vector)
id_rotation = np.eye(3, 3)  # make identity matrix

pos_in_time = np.zeros((len(times), 3))  # make array fill w 0
for k in np.arange(0, len(times)):
    xyz = _apply_affine(init_pos, id_rotation, translations[k])
    pos_in_time[k] = xyz
    
# find eclidean distance for each trans
euc_dist = np.zeros(len(times) - 1)
for ii in np.arange(0, pos_in_time.shape[0] - 1):
    euc_dist[ii] = euclidean(init_pos, pos_in_time[ii])
    
# using identity ROTS
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_in_time[:, 0], pos_in_time[:, 1], pos_in_time[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# sanity check TRANS only
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(translations[:, 0], translations[:, 1], translations[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# plot displacement
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(times[:-1], euc_dist)
ax.set_xlabel('Time')
ax.set_ylabel('Displacement')

# histogram of displacement data
mu = euc_dist.mean() * 1e3
sigma = euc_dist.std() * 1e3
fig = plt.figure()
n, bins, patches = plt.hist(euc_dist * 1e3, 100, normed=True, facecolor='green', alpha=0.75)
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r-', linewidth=2)
plt.xlabel('Displacement (mm)')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ head\ positions:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
plt.axis([min(bins), max(bins), min(n), max(n)])
plt.grid(True)
plt.ioff()
plt.show()
