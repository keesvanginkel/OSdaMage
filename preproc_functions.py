import os 
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
from shapely.geometry import MultiPolygon
from geopy.distance import vincenty
logging.basicConfig(filename='OSM_extracts.log',level=logging.INFO)

def poly_files_europe(data_path, NUTS_shape):
    
    """
    Based on Elco's poly_files, but now for Europe 
    """
    
    NUTS_poly = gpd.read_file(NUTS_shape)
    
    # remove all remote overseas areas
    remote_regions = ["ES703","ES704","ES705","ES706","ES707","ES708","ES709",
                      "FRY10","FRY20","FRY30","FRY40","FRY50",
                      "PT300","PT200"]
    NUTS_poly = NUTS_poly[~NUTS_poly['NUTS_ID'].isin(remote_regions)]
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
            f = open(data_path + "/" + nuts_id +'.poly', 'w')
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
        except:
            print(f['NUTS_ID']) 
            
def clip_osm(osm_convert_path,planet_path,area_poly,area_pbf):
    """ Clip the an area osm file from the larger continent (or planet) file and save to a new osm.pbf file. 
    This is much faster compared to clipping the osm.pbf file while extracting through ogr2ogr.
    
    This function uses the osmconvert tool, which can be found at http://wiki.openstreetmap.org/wiki/Osmconvert. 
    
    Either add the directory where this executable is located to your environmental variables or just put it in the 'scripts' directory.
    
    Arguments:
    
        osm_convert_path: path string to the palce where the osm_convert executable is located
        
        planet_path: path string to the .planet files containing the OSM Europe or OSM world file from which you want to crop
        
        area_poly: path string to the .poly file, made through the 'create_poly_files' function.
        
        area_pbf: path string indicating the final output dir and output name of the new .osm.pbf file.
        
    Returns:
    
        a clipped .osm.pbf file (saved as area_pbf .osm.pbf)
    """ 
    print('{} started!'.format(area_pbf))

    try: 
        if (os.path.exists(area_pbf) is not True):
            os.system('{}  {} -B={} --complete-ways -o={}'.format(osm_convert_path,planet_path,area_poly,area_pbf))
        print('{} finished!'.format(area_pbf))

    except:
        print('{} did not finish!'.format(area_pbf))
        
def clip_osm_multi(dirs):
    """ Clip the an area osm file from the larger continent (or planet) file and save to a new osm.pbf file. 
    This is much faster compared to clipping the osm.pbf file while extracting through ogr2ogr.
    
    This function uses the osmconvert tool, which can be found at http://wiki.openstreetmap.org/wiki/Osmconvert. 
    
    Either add the directory where this executable is located to your environmental variables or just put it in the 'scripts' directory.
    
    Arguments [contained in dirs to enable multiprocessing]:
    
        dirs[0] = osm_convert_path: path string to the palce where the osm_convert executable is located
        
        dirs[1] = planet_path: path string to the .planet files containing the OSM Europe or OSM world file from which you want to crop
        
        dirs[2] = area_poly: path string to the .poly file, made through the 'create_poly_files' function.
        
        dirs[3] = area_pbf: path string indicating the final output dir and output name of the new .osm.pbf file.
        
    Returns:
    
        a clipped .osm.pbf file (saved as area_pbf .osm.pbf)
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
        
        # -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:57:52 2019

@author: cenv0574
"""



def remove_tiny_shapes(x,regionalized=False):
    """This function will remove the small shapes of multipolygons. Will reduce the size of the file.
    
    Arguments:
        x {feature} -- a geometry feature (Polygon) to simplify. Countries which are very 
        large will see larger (unhabitated) islands being removed.
    
    Keyword Arguments:
        regionalized {bool} -- set to True to allow for different threshold settings (default: {False})
    """

    if (x.continent == 'Oceania') | (x.continent == 'Central-America'):
        return x.geometry
    
    if x.geometry.geom_type == 'Polygon':
        return x.geometry
    elif x.geometry.geom_type == 'MultiPolygon':
        
        if regionalized == False:
            area1 = 0.1
            area2 = 250
                
        elif regionalized == True:
            area1 = 0.01
            area2 = 50           

        # dont remove shapes if total area is already very small
        if x.geometry.area < area1:
            return x.geometry
        # remove bigger shapes if country is really big

        if x['GID_0'] in ['CHL','IDN']:
            threshold = 0.01
        elif x['GID_0'] in ['RUS','GRL','CAN','USA']:
            if regionalized == True:
                threshold = 0.01
            else:
                threshold = 0.01

        elif x.geometry.area > area2:
            threshold = 0.1
        else:
            threshold = 0.001

        # save remaining polygons as new multipolygon for the specific country
        new_geom = []
        for y in x.geometry:
            if y.area > threshold:
                new_geom.append(y)
        
        return MultiPolygon(new_geom)

def global_shapefiles(regionalized=False):
    """ 
    This function will simplify shapes and add necessary columns, to make further processing more quickly
    
    For now, we will make use of the latest GADM data: https://gadm.org/download_world.html

    Keyword Arguments:
        regionalized {bool} -- When set to True, this will also create the global_regions.shp file. (default: {False})
    """

    data_path = load_config()['paths']['data']
   
    # path to country GADM file
    if regionalized == False:
        
        # load country file
        country_gadm_path = os.path.join(data_path,'GADM','gadm34_0.shp')
        gadm_level0 = gpd.read_file(country_gadm_path)
    
        # remove antarctica, no roads there anyways
        gadm_level0 = gadm_level0.loc[~gadm_level0['NAME_0'].isin(['Antarctica'])]
        
        # remove tiny shapes to reduce size substantially
        gadm_level0['geometry'] =   gadm_level0.apply(remove_tiny_shapes,axis=1)
    
        # simplify geometries
        gadm_level0['geometry'] = gadm_level0.simplify(tolerance = 0.005, preserve_topology=True).buffer(0.01).simplify(tolerance = 0.005, preserve_topology=True)
        
        # add additional info
        glob_info_path = os.path.join(data_path,'input_data','global_information.xlsx')
        load_glob_info = pd.read_excel(glob_info_path)
        
        gadm_level0 = gadm_level0.merge(load_glob_info,left_on='GID_0',right_on='ISO_3digit')
   
        #save to new country file
        glob_ctry_path = os.path.join(data_path,'input_data','global_countries.shp')
        gadm_level0.to_file(glob_ctry_path)
          
    else:

        # this is dependent on the country file, so check whether that one is already created:
        glob_ctry_path = os.path.join(data_path,'input_data','global_countries.shp')
        if os.path.exists(glob_ctry_path):
            gadm_level0 = gpd.read_file(os.path.join(data_path,'input_data','global_countries.shp'))
        else:
            print('ERROR: You need to create the country file first')   
            return None
        
    # load region file
        region_gadm_path = os.path.join(data_path,'GADM','gadm34_1.shp')
        gadm_level1 = gpd.read_file(region_gadm_path)
       
        # remove tiny shapes to reduce size substantially
        gadm_level1['geometry'] =   gadm_level1.apply(remove_tiny_shapes,axis=1)
    
        # simplify geometries
        gadm_level1['geometry'] = gadm_level1.simplify(tolerance = 0.005, preserve_topology=True).buffer(0.01).simplify(tolerance = 0.005, preserve_topology=True)
        
        # add additional info
        glob_info_path = os.path.join(data_path,'input_data','global_information.xlsx')
        load_glob_info = pd.read_excel(glob_info_path)
        
        gadm_level1 = gadm_level1.merge(load_glob_info,left_on='GID_0',right_on='ISO_3digit')
        gadm_level1.rename(columns={'coordinates':'coordinate'}, inplace=True)
    
        # add some missing geometries from countries with no subregions
        get_missing_countries = list(set(list(gadm_level0.GID_0.unique())).difference(list(gadm_level1.GID_0.unique())))
        
        mis_country = gadm_level0.loc[gadm_level0['GID_0'].isin(get_missing_countries)]
        mis_country['GID_1'] = mis_country['GID_0']+'_'+str(0)+'_'+str(1)
    
        gadm_level1 = gpd.GeoDataFrame( pd.concat( [gadm_level1,mis_country] ,ignore_index=True) )
        gadm_level1.reset_index(drop=True,inplace=True)
       
        #save to new country file
        gadm_level1.to_file(os.path.join(data_path,'input_data','global_regions_v2.shp'))

def poly_files(data_path,global_shape,save_shapefile=False,regionalized=False):

    """
    This function will create the .poly files from the world shapefile. These
    .poly files are used to extract data from the openstreetmap files.
    
    This function is adapted from the OSMPoly function in QGIS.
    
    Arguments:
        base_path: base path to location of all files.
        
        global_shape: exact path to the global shapefile used to create the poly files.
        
        save_shape_file: when True, the new shapefile with the countries that we include in this analysis will be saved.       
    
    Returns:
        .poly file for each country in a new dir in the working directory.
    """     
    
# =============================================================================
#     """ Create output dir for .poly files if it is doesnt exist yet"""
# =============================================================================
    poly_dir = os.path.join(data_path,'country_poly_files')
    
    if regionalized == True:
        poly_dir = os.path.join(data_path,'regional_poly_files')
    
    if not os.path.exists(poly_dir):
        os.makedirs(poly_dir)

# =============================================================================
#   """Load country shapes and country list and only keep the required countries"""
# =============================================================================
    wb_poly = gpd.read_file(global_shape)
    
    # filter polygon file
#    if regionalized == True:
#        wb_poly = wb_poly.loc[wb_poly['GID_0'] != '-']
#        wb_poly = wb_poly.loc[wb_poly['TYPE_1'] != 'Water body']
#   else:
#       wb_poly = wb_poly.loc[wb_poly['ISO_3digit'] != '-']
   
    wb_poly.crs = {'init' :'epsg:4326'}

    # and save the new country shapefile if requested
    if save_shapefile == True:
        wb_poly.to_file(wb_poly_out)
    
    # we need to simplify the country shapes a bit. If the polygon is too diffcult,
    # osmconvert cannot handle it.
#    wb_poly['geometry'] = wb_poly.simplify(tolerance = 0.1, preserve_topology=False)

# =============================================================================
#   """ The important part of this function: create .poly files to clip the country 
#   data from the openstreetmap file """    
# =============================================================================
    num = 0
    # iterate over the counties (rows) in the world shapefile
    for f in wb_poly.iterrows():
        f = f[1]
        num = num + 1
        geom=f.geometry

        try:
            # this will create a list of the different subpolygons
            if geom.geom_type == 'MultiPolygon':
                polygons = geom
            
            # the list will be lenght 1 if it is just one polygon
            elif geom.geom_type == 'Polygon':
                polygons = [geom]

            # define the name of the output file, based on the ISO3 code
            ctry = f['GID_0']
            if regionalized == True:
                attr=f['GID_2']
            else:
                attr=f['GID_0']
            
            # start writing the .poly file
            f = open(poly_dir + "/" + attr +'.poly', 'w')
            f.write(attr + "\n")

            i = 0
            
            # loop over the different polygons, get their exterior and write the 
            # coordinates of the ring to the .poly file
            for polygon in polygons:
    #
    #            if ctry == 'CAN':
    #                dist = vincenty(polygon.centroid.coords[:1][0], (-88.68,79.14), ellipsoid='WGS-84').kilometers
    #                if dist < 500:
    #                    continue

                if ctry == 'RUS':
                    dist = vincenty(tuple(reversed(polygon.centroid.coords[:1][0])), (58.89,82.26), ellipsoid='WGS-84').kilometers
                    if dist < 500:
                        continue
                    
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
        except:
            print(f['GID_2'])
            
