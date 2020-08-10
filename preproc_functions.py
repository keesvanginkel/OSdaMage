"""
Preprocessing functions of OSdaMage 1.0.

Contains the preprocessing functions required for running the OSdaMage model. The functions are called from a Jupyter Notebook 'Preproc_split_OSM.ipynb'

This code is maintained on a GitHub repository: github.com/keesvanginkel/OSdaMage

@author: Elco Koks and Kees van ginkel
"""

import geopandas as gpd
import logging
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
from shapely.geometry import MultiPolygon
from geopy.distance import vincenty

logging.basicConfig(filename='OSM_extracts.log',level=logging.INFO)

def poly_files_europe(out_path, NUTS_shape,filter_out):
    
    """
    This function will create the .poly files from the Europe shapefile.
    .poly files are used to extract data from the openstreetmap files.
    
    This function is adapted from the OSMPoly function in QGIS, and Elco Koks GMTRA model.
    
    Arguments:
        *out_path* (string): path to the directory where the .poly files should be written
        *NUTS_shape* (string) : path to the NUTS-3 shapefile (CRS=EPSG:3035)
        *filter_out* (list of strings): names of NUTS-3 regions not to include in the analysis
    
    Returns:
        .poly file for each country in a new dir in the working directory (CRS=WGS:84).
    """   
    
    NUTS_poly = gpd.read_file(NUTS_shape)
    
    #Remove regions that are to be filtered out
    NUTS_poly = NUTS_poly[~NUTS_poly['NUTS_ID'].isin(filter_out)]
    NUTS_poly = NUTS_poly.to_crs(epsg=4326) #Change into the WGS84 = EPSG4326 coordinate system of OSM.

    num = 0
    # iterate over the counties (rows) in the Europe shapefile
    for f in NUTS_poly.iterrows():
        f = f[1]
        num = num + 1
        geom=f.geometry

        try:
            # this will create a list of the different subpolygons
            if geom.geom_type == 'MultiPolygon':
                polygons = geom

            # the list will be length 1 if it is just one polygon
            elif geom.geom_type == 'Polygon':
                polygons = [geom]

            # define the name of the output file, based on the NUTS_ID
            nuts_id = f['NUTS_ID']

            # start writing the .poly file
            f = open(out_path + "/" + nuts_id +'.poly', 'w')
            f.write(nuts_id + "\n")

            i = 0

            # loop over the different polygons, get their exterior and write the 
            # coordinates of the ring to the .poly file
            for polygon in polygons:

                polygon = np.array(polygon.exterior)

                j = 0
                f.write(str(i) + "\n")

                for ring in polygon:
                    j = j + 1
                    f.write("    " + str(ring[0]) + "     " + str(ring[1]) +"\n")

                i = i + 1
                # close the ring of one subpolygon if done
                f.write("END" +"\n")

            # close the file when done
            f.write("END" +"\n")
            f.close()
        except Exception as e:
            print("Exception {} for {}" .format(e,f['NUTS_ID'])) 
            
def clip_osm_multi(dirs): #called from the Preproc_split_OSM file
    """ Clip the an area osm file from the larger continent (or planet) file and save to a new osm.pbf file. 
    This is much faster compared to clipping the osm.pbf file while extracting through ogr2ogr.
    
    This function uses the osmconvert tool, which can be found at http://wiki.openstreetmap.org/wiki/Osmconvert. 
    
    Either add the directory where this executable is located to your environmental variables or just put it in the 'scripts' directory.
    
    Arguments (stored in a list to enable multiprocessing):
        *dirs[0] = osm_convert_path* (string): path to the osm_convert executable
        *dirs[1] = planet_path* (string): path  to the .planet file
        *dirs[2] = area_poly* (string): path to the .poly file, made by create_poly_files_europe()
        *dirs[3] = area_pbf* (string): output directory
        
    Returns:
        *region.osm.pbf* (os.pbf file) : the clipped (output) osm.pbf file
    """ 
    
    osm_convert_path = dirs[0]
    planet_path = dirs[1]
    area_poly = dirs[2]
    area_pbf = dirs[3]
    
    print('{} started!'.format(area_pbf))
    logging.info('{} started!'.format(area_pbf))
    
    try: 
        if (os.path.exists(area_pbf) is not True):
            os.system('{}  {} -B={} --complete-ways --hash-memory=500 -o={}'.format(osm_convert_path,planet_path,area_poly,area_pbf))
            print('{} finished!'.format(area_pbf))
            logging.info('{} finished!'.format(area_pbf))
        else:
            print('{} already exists'.format(area_pbf))
            logging.info('{} already exists'.format(area_pbf))

    except Exception as e:
        logging.error('{} did not finish because of {}'.format(area_pbf,str(e)))