# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:25:11 2018

@author: cenv0574
"""

import os
import json
from geopy.distance import vincenty
from boltons.iterutils import pairwise
import geopandas as gpd
import shapely.ops
import pandas as pd
import shapely.wkt
import shutil

def load_config():
    """Read config.json
    """
#    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

def clean_dir(dirpath):
    """"This function can be used to fully clear a directory
    
    Arguments:
        dirpath {string} -- path to directory to be cleared from files
    """
    
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
            
def remove_files(dirpath,startname):
    """This function can be used to delete specific files from a directory. In 
    general this function is used to clean country files from the 'calc' directory
    
    Arguments:
        dirpath {string} -- path to directory in which the files should be removed
        startname {string} -- the substring to be searched for in the files
    """
    for fname in os.listdir(dirpath):
        if fname.startswith(startname):
            os.remove(os.path.join(dirpath, fname))

def create_folder_structure(data_path,regionalized=True):
    """Create the directory structure for the output
    
    Arguments:
        base_path {string} -- path to directory where folder structure should be created 
    
    Keyword Arguments:
        regionalized {bool} -- specify whether also the folders for a regionalized analyse should be created (default: {True})
    """
    
    data_path = load_config()['paths']['data']
    
    # create calc dir
    calc_dir = os.path.join(data_path,'calc')
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)

    # create country osm dir
    dir_osm = os.path.join(data_path,'country_osm')
    if not os.path.exists(dir_osm):
        os.makedirs(dir_osm)    
        
    if regionalized == True:
        dir_osm_region = os.path.join(data_path,'region_osm')
        if not os.path.exists(dir_osm_region):
            os.makedirs(dir_osm_region)          
        
    # create output dirs for country level analysis
    dir_out = os.path.join(data_path,'country_data')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
       
    # create dir and output file for continent outputs
    continent_dir_high = os.path.join(data_path,'continental_highway')
    if not os.path.exists(continent_dir_high):
        os.makedirs(continent_dir_high)
    continent_dir_rail = os.path.join(data_path,'continental_railway')
    if not os.path.exists(continent_dir_rail):
        os.makedirs(continent_dir_rail)
    continent_dir_avi = os.path.join(data_path,'continental_aviation')
    if not os.path.exists(continent_dir_avi):
        os.makedirs(continent_dir_avi)     
    continent_dir_shi = os.path.join(data_path,'continental_shipping')
    if not os.path.exists(continent_dir_shi):
        os.makedirs(continent_dir_shi)
    continent_dir_mm = os.path.join(data_path,'continental_multimodal')
    if not os.path.exists(continent_dir_mm):
        os.makedirs(continent_dir_mm)        

def simplify_paths(x):
    """simpilify the geometry of the specific geometric feature. This will 
        reduce the calculation times quite a bit.
    
    Arguments:
        x {shapely geometry} -- a shapely geometry to be simplified. 
    """
    return x.simplify(tolerance=0.0001, preserve_topology=True)

def feather_to_gpd(file_name,save_shapefile=False):
    """Read a pandas dataframe saved as a feather file (.ft) and convert to a 
    geodatabase. 
    
    Arguments:
        file_name {string} -- full path to feather file to load
    
    Keyword Arguments:
        save_shapefile {bool} -- Save the created GeoDatabase as shapefile (default: {False})
    
    Returns:
        Geopandas geodataframe of the .feather file
    """

    # load feather file, convert geometry to normal linestrings again and 
    # fix some othe rsmall issues before saving it to a shapefile
    gpd_in = gpd.GeoDataFrame(pd.read_feather(file_name))
    gpd_in['geometry'] = gpd_in['geometry'].apply(shapely.wkb.loads)
    gpd_in.crs = {'init' :'epsg:4326'}
    gpd_in['infra_type'] = gpd_in['infra_type'].astype(str)
    gpd_in['country'] = gpd_in['country'].astype(str)

    gpd_in = gpd.GeoDataFrame(gpd_in,geometry='geometry')
    
    if save_shapefile == True:
        # set new file name
        output_name =  file_name[:-3]+'.shp'

        gpd_in.to_file(output_name)

    return gpd_in

def line_length(line, ellipsoid='WGS-84',shipping=True):
    """Length of a line in meters, given in geographic coordinates

    Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Arguments:
        line {Shapely LineString} -- a shapely LineString object with WGS-84 coordinates
        ellipsoid {String} -- string name of an ellipsoid that `geopy` understands (see
            http://geopy.readthedocs.io/en/latest/#module-geopy.distance)

    Returns:
        Length of line in meters
    """
    if shipping == True:
        if line.geometryType() == 'MultiLineString':
            return sum(line_length(segment) for segment in line)
    
        return sum(
            vincenty(tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid).kilometers
            for a, b in pairwise(line.coords)
    )

    else:
        if line.geometryType() == 'MultiLineString':
            return sum(line_length(segment) for segment in line)
    
        return sum(
            vincenty(a, b, ellipsoid=ellipsoid).kilometers
            for a, b in pairwise(line.coords)
    )

    
