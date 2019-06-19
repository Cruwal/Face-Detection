import numpy
import scipy
import imageio
from scipy import ndimage
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
'''
im = scipy.misc.imread('BioID_0194.pgm')
im = im.astype('int32')
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag = numpy.hypot(dx, dy)  # magnitude
mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
# Ploting image
plt.imshow(mag)
plt.show()
scipy.misc.imsave('sobel.jpg', mag)
'''
image = imageio.imread('BioID_0194.pgm')
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=False)

plt.imshow(hog_image)
plt.show()