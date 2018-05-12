#!/usr/bin/python3

import os
import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

frame_files = os.listdir('frames')
print('loading {} frames...'.format(len(frame_files)))
frames = []
for filename in sorted(frame_files):
   img = Image.open('frames/' + filename)
   frames += [np.array(img)]
    
frames = np.array(frames).astype('float32')

# take red and suppress other colors
red = np.maximum(0, frames[..., 0] - frames[..., 1] - frames[..., 2])
red = red > 20

positions = []
for i in range(len(frames)):
  frame = red[i] # gaussian_filter(frames[i], 8)
  p = np.median(np.argwhere(frame), 0).astype('int')
  positions += [p]
positions = np.array(positions)

# iir filter
a = 0.25  # filter parameter
pos_filtered = [positions[0]]
for t in range(1, len(positions)):
  pos_filtered += [(a * pos_filtered[t-1] + (1-a) * positions[t]).astype('int')]

f = frames[-1]
for i in range(len(frames)):
  p = positions[i]
  f[p[0]-1:p[0]+1, p[1]-1: p[1]+1,0] = 255

  p = pos_filtered[i]
  f[p[0]-1:p[0]+1, p[1]-1: p[1]+1,1] = 255

Image.fromarray(f.astype('uint8')).show()
#Image.fromarray(red[0].astype('uint8')*255).show()

R=np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],[np.sin(np.pi/2), np.cos(np.pi/2)]])

positions = np.dot(positions,R)
pos_filtered = np.dot(pos_filtered,R)

np.savetxt('target_trajectory.dat', pos_filtered, fmt="%12.6f %12.6f")
