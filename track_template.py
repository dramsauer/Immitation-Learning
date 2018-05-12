#!/usr/bin/python3

import os
import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

img = Image.open('frames/00000.jpg')
img.show()


## open images files from a directory in a loop
#frame_files = os.listdir('frames')
#for filename in sorted(frame_files):
  #img = Image.open('frames/' + filename)

#np.array(img)  # convert PIL.Image to numpy array

#np.maximum(0, a)  # element-wise maximum

#np.median(a, 1)  # medium of array a along dimension 1

#np.mean(a, 1)   # mean of array a along dimension 1

#np.argwhere(c)  # position of the array elements that satisfy the condition c

#gaussian_filter(a, 3)  #  image smoothing; gaussian filter of array a (with sigma=3)

#a.astype('float32')  # convert array a to float

#a.astype('uint8')  # convert array a to unsigned int8

#img = Image.fromarray(a)  # convert array a to PIL.Image object

#img.show()  # show image img (img must be uint8)

## trajectory smoothing
#a = 0.25  # filter parameter
#trajectory_filtered = [trajectory[0]]
#for t in range(1, len(trajectory)):
#  trajectory_filtered += [(a * trajectory_filtered[t-1] + (1-a) *trajectory[t]).astype('int')]
  
#R=np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],[np.sin(np.pi/2), np.cos(np.pi/2)]]) 2D rotation matrix of 90 degrees

#A = np.dot(A,R) #matrix rotation 90 degress

#np.savetxt('file_name', A, fmt="%12.6f %12.6f") #save nx2 matrix into a file
