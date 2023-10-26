### Script to check the labels provided by the paper 
import os
import sys
import argparse
import glob


import numpy as np
from osgeo import gdal, gdal_array, ogr
import pandas as pd
import matplotlib.pyplot as plt


path = '/mnt/data/OpenSentinelMap/osm_sentinel_imagery/43QEU/82543044/'

file = 'S2A_43QEU_20201018_0_L2A.npz'

data = np.load(path+file)

list(data.keys())
list(data.values())
list(data.items())


b10m = data.get('gsd_10')
b20m = data.get('gsd_20')
b60m = data.get('gsd_60')
bscl = data.get('scl')
bad_perc = data.get('bad_percent')


fig = plt.figure()
im2 = plt.title("Sub-window Image")
im2 = plt.xlabel("Projection angle (deg)")
im2 = plt.ylabel("Projection position (pixels)")
#im2 = plt.imshow(np.abs(test), cmap=plt.cm.viridis,
#            aspect='auto')
#im2 = plt.xlim(150, 220)
#im2 = plt.ylim(20, 60)
cmap = plt.get_cmap('jet', 7)
im2 = plt.imshow(bscl , cmap=cmap,  
                             aspect='auto', interpolation='none')
cbar = fig.colorbar(im2, extend='both')


import glob
import os.path 
import imageio

png_path = '/data/OpenSentinelMap/osm_label_images/osm_label_images_v10/'
files = [f for f in glob.glob(png_path + '***/***.png', recursive = True)]
#png_path = '/data/OpenSentinelMap/osm_label_images/osm_label_images_v10/52UCG/119316571.png'
#119316572.png'

#cmap = plt.cm.jet
#colors = cmap(np.arange(cmap.N))
#plt.cm.get_cmap('Blues', 6)
for file in files: 

    im = imageio.imread(file)
    #imageio.imread(im[:,:,0])
    png_im = np.array(im)

    plt.imshow(im[:,:,0], cmap=plt.cm.get_cmap('magma', 10),  
                        aspect='auto', interpolation='none')
    #plt.clim(0, 10);
    plt.show()

# from PIL import Image
# import numpy as np

# im_frame = Image.open(png_path )
# np_frame = np.array(im_frame.getdata())
