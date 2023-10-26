import os
import sys
import argparse
import glob


import numpy as np
from osgeo import gdal, gdal_array, ogr
import pandas as pd
from scipy import ndimage
from sklearn import cluster
import math 
import matplotlib.pyplot as plt
import cv2 as cv




def read_data(inputPath):
    global geoMatrix, rows, cols, proj, data
    bands = []
    for i in [["R10m", "B02"], ["R10m", "B03"], ["R10m", "B04"], ["R10m", "B08"], ["R20m", "B8A"], ["R20m", "B11"], ["R20m", "SCL"]]:
        # Open the input dataset for the selected band
        print("Retrieving file" + os.path.join(inputPath, i[0], "*" + i[1] + "*.jp2"))
        f = glob.glob(os.path.join(inputPath, i[0], "*" + i[1] + "*.jp2"))[0]
        print("Opening band {} located in file {}...".format(i, f))
        dataset = gdal.Open(f, gdal.GA_ReadOnly)

        #Store important information along the way
        if i[1] == "B02":
            geoMatrix = dataset.GetGeoTransform()
            rows = dataset.RasterYSize
            cols = dataset.RasterXSize
            proj = dataset.GetProjection()

         # Get data
        band = dataset.GetRasterBand(1)
        label = band.GetDescription()
        if i[1] == "SCL": # The SCL file needs to be preserved as is
            data = band.ReadAsArray().astype(np.uint8)
        else: # We need the reflectance of the other bands
            data = band.ReadAsArray().astype(np.float32) / 10000.0
        bands.append(data)
        print("Band {} (Description: {}) loaded!".format(i, label))

    b02, b03, b04, b08, b8a, b11, scl = bands
    print("All bands successfuly loaded!")

    # Resize array that need it

    print("Resizing bands...")
    b8a = ndimage.zoom(b8a, (b04.shape[1] / b8a.shape[1], b04.shape[0] / b8a.shape[0]), order=0)
    b11 = ndimage.zoom(b11, (b04.shape[1] / b11.shape[1], b04.shape[0] / b11.shape[0]), order=0)
    scl = ndimage.zoom(scl, (b04.shape[1] / scl.shape[1], b04.shape[0] / scl.shape[0]), order=0)
    print("Bands successfuly resized!")


    """ INDEX/MASKS COMPUTATION """
    print("Computing masks and special indices...")
    gndvi  =    (b08-b03) / (b08+b03)

    landmask    =   (scl >= 3) & (scl <= 5) & (gndvi > 0)
    cloudmask   =   (scl >= 8) & (scl <= 10)
    bad         =   ((cloudmask == 1) | (landmask == 1))

    print("Masks computed!")
    
    #b02[bad]=np.nan
    #b03[bad]=np.nan
    #b04[bad]=np.nan
    #b08[bad]=np.nan
    #b8a[bad]=np.nan
    #b11[bad]=np.nan
    
    data= b02, b03, b04, b08, b8a, b11
    print("Data to use ready!") 
    return data
	
def main():
    """ PARSING ARGUMENTS """
        
    # Create argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="The path to the image data")
    args = vars(ap.parse_args())

    # Parse arguments
    print("Parsing arguments...")
    inputPath = args["input"]
	
	
	data = read_data(inputPath)
	# band 2 == data[0]
	# band 3 = data[1]
	# band 4 = data[2]
	# band 8 = data[3]
	# band 8a = data[4]
	# band 11 = data[5]
	



if __name__ == "__main__":
    main()	
	
	
	