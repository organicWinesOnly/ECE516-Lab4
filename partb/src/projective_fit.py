""" projective_fit.py

    This file dechirps and upchirps two images so tht they match the
    non-chirping version.
"""

import cv2 as cv
from skimage.transform import *
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Polygon

non_chirp = imread("../data/non_chirp.jpg")
upchirp = imread("../data/upchirp.jpg")
downchirp = imread("../data/downchirp.jpg")

non_chirp = pyramid_reduce(non_chirp)
non_chirp = pyramid_reduce(non_chirp)
upchirp = pyramid_reduce(upchirp)
upchirp = pyramid_reduce(upchirp)
downchirp = pyramid_reduce(downchirp)
downchirp = pyramid_reduce(downchirp)
src = np.array([[2, 226],[2, 252],  [378, 247], [377, 226]]) 
dst_up = np.array([[3,252],[3,285],[381,263],[381,249]])
dst_down = np.array([[3,251],[2,266],[376,265],[378,237]])
Pu = estimate_transform('projective', src, dst_up)
Pd = estimate_transform('projective', src, dst_down)

upchirp_t = warp(upchirp, Pu)
downchirp_t = warp(downchirp, Pd)

plt.figure()
ax = plt.subplot(231)
ax.set_title("Non-chirping image")
ax.imshow(non_chirp, cmap=cm.get_cmap('gray'))
p = Polygon(src, fill=False, color='red')
ax.add_patch(p)

bx = plt.subplot(232)
bx.set_title("Up-chirping image")
bx.imshow(upchirp, cmap=cm.get_cmap('gray'))
p = Polygon(dst_up, fill=False, color='red')
bx.add_patch(p)

cx = plt.subplot(233)
cx.set_title("Down-chirping image")
cx.imshow(downchirp, cmap=cm.get_cmap('gray'))
p = Polygon(dst_down, fill=False, color='red')
cx.add_patch(p)

dx = plt.subplot(223)
dx.set_title("Transformed up-chirping image")
dx.imshow(upchirp_t, cmap=cm.get_cmap('gray'))

ex = plt.subplot(224)
ex.set_title("Transformed down-chirping image")
ex.imshow(downchirp_t, cmap=cm.get_cmap('gray'))
plt.show()
