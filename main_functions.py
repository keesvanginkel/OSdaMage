# -*- coding: utf-8 -*-
"""
Main functions of OSdaMage 1.0.

Contains the main functionality of the OSdaMage model. The functions are called from a Jupyter Notebook 'Main_multi.ipynb', 
starting from the function region_loss_estimation.

This code is maintained on a GitHub repository: github.com/keesvanginkel/OSdaMage

@author: Kees van Ginkel and Elco Koks
"""


from collections import defaultdict, OrderedDict
import geopandas as gpd
from natsort import natsorted
import numpy as np
import ogr
import os
import pandas as pd
from pdb import set_trace #for debugging
import pickle
from random import shuffle
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import shapely
import sys
from shapely.geometry import mapping
import time as time
from tqdm import tqdm

from utils_functions import load_config,line_length

def region_loss_estimation(region, **kwargs):
    """
    Coordinates the loss estimation for the region.
    
    Arguments:

        *region* (string) -- NUTS3 code of region to consider.
    
    Returns:

        *csv file* (csv file) -- All inundated road segments in region, each row is segment, columns:
            osm_id (integer) : OSM ID
            infra_type (string) : equals OSM highway key
            geometry (LINESTRING): road line geometry (simplified version of OSM shape)
            lanes (integer): # lanes of road segment (from OSM or estimated based on median of country)
            bridge (str): boolean indicating if it is a bridge or not
            lit (str): boolean indicating if lighting is present
            length (float): length of road segment in m
            road_type (str): mapped road type (e.g. motorway, trunk, ... , track) for use by damage cal.
            length_rp10 ... rp500 (float): length of the inundation section per hazard RP in m
            val_rp10 ... rp500 (float): average depth over inundated section per hazard RP
            NUTS-3 ... NUTS-0 (str): regional NUTS-ID of the segment
            dam_CX...rpXX (tuple): containing (min, 25%, 50%, 75%, max) of damage estimate (Euros) for damage curve X 
        
        *pickle* -- contains pd.DataFrame similar to csv file: for fast loading
    
    """   
    from postproc_functions import NUTS_down 

    try:
          
        #LOAD DATA PATHS - configured in the config.json file
        osm_path = load_config()['paths']['osm_data'] #this is where the osm-extracts are located
        input_path = load_config()['paths']['input_data'] #this is where the other inputs (such as damage curves) are located     
        hazard_path =  load_config()['paths']['hazard_data'] #this is where the inundation raster are located
        output_path = load_config()['paths']['output'] #this is where the results are to be stored

        #CREATE A LOG FILE OR TAKE THE FILE FOM THE KEYWORD ARGUMENTS
        log_file = kwargs.get('log_file', None)
        if log_file is None:
            log_file = os.path.join(output_path,"region_loss_estimation_log_{}.txt".format(os.getenv('COMPUTERNAME')))

        if log_file is not None: #write to log file
            file = open(log_file, mode="a")
            file.write("\n\nRunning region_loss_estimation for region: {} at time: {}\n".format(region,
                                time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
            file.close()

        # SKIP IF REGION IS ALREADY FINISHED BY CHECKING IF OUTPUT FILE IS ALREADY CREATED
        if os.path.exists(os.path.join(output_path,'{}.csv'.format(region))): 
            print('{} already finished!'.format(region))
            return None

        # IMPORT FLOOD CURVES AND DAMAGE DATA
        #Load the Excel file containing the OSM mapping and damage curves
        map_dam_curves = load_config()['filenames']['map_dam_curves'] 
        interpolators = import_flood_curves(filename = map_dam_curves, sheet_name='All_curves', usecols="B:O")
        dict_max_damages = import_damage(map_dam_curves,"Max_damages",usecols="C:E")
        max_damages_HZ = load_HZ_max_dam(map_dam_curves,"Huizinga_max_dam","A:G")

        # LOAD NUTS REGIONS SHAPEFILE
        NUTS_regions = gpd.read_file(os.path.join(input_path, load_config()['filenames']['NUTS3-shape']))

        # EXTRACT ROADS FROM OSM FOR THE REGION
        road_gdf = fetch_roads(osm_path,region,log_file=os.path.join(output_path,'fetch_roads_log_{}.txt'.format(os.getenv('COMPUTERNAME'))))
        
        # CLEANUP THE ROAD EXTRACTION
        road_gdf = cleanup_fetch_roads(road_gdf, region)

        # CALCULATE LINE LENGTH, SIMPLIFY GEOMETRY, MAP ROADS BASED ON EXCEL CLASSIFICATION
        road_gdf['length'] = road_gdf.geometry.apply(line_length)
        road_gdf.geometry = road_gdf.geometry.simplify(tolerance=0.00005) #about 0.001 = 100 m; 0.00001 = 1 m
        road_dict = map_roads(map_dam_curves,'Mapping')
        road_gdf['road_type'] = road_gdf.infra_type.apply(lambda x: road_dict[x]) #add a new column 'road_type' with less categories
        
        # GET GEOMETRY OUTLINE OF REGION
        geometry = NUTS_regions['geometry'].loc[NUTS_regions.NUTS_ID == region].values[0]
               
        # CREATE DATAFRAME WITH VECTORIZED HAZARD DATA FROM INPUT TIFFS
        hzd_path = os.path.join(hazard_path) 
        hzd_list = natsorted([os.path.join(hzd_path, x) for x in os.listdir(hzd_path) if x.endswith(".tif")])
        hzd_names = ['rp10','rp20','rp50','rp100','rp200','rp500']
        
        hzds_data = create_hzd_df(geometry,hzd_list,hzd_names) #both the geometry and the hzd maps are still in EPSG3035
        hzds_data = hzds_data.to_crs({'init': 'epsg:4326'}) #convert to WGS84=EPSG4326 of OSM.
        
        # PERFORM INTERSECTION BETWEEN ROAD SEGMENTS AND HAZARD MAPS
        for iter_,hzd_name in enumerate(hzd_names):
            
            try:
                hzd_region = hzds_data.loc[hzds_data.hazard == hzd_name]
                hzd_region.reset_index(inplace=True,drop=True)
            except:
                hzd_region == pd.DataFrame(columns=['hazard'])
            
            if len(hzd_region) == 0:
                road_gdf['length_{}'.format(hzd_name)] = 0
                road_gdf['val_{}'.format(hzd_name)] = 0
                continue
            
            hzd_reg_sindex = hzd_region.sindex
            tqdm.pandas(desc=hzd_name+'_'+region) 
            inb = road_gdf.progress_apply(lambda x: intersect_hazard(x,hzd_reg_sindex,hzd_region),axis=1).copy()
            inb = inb.apply(pd.Series)
            inb.columns = ['geometry','val_{}'.format(hzd_name)]
            inb['length_{}'.format(hzd_name)] = inb.geometry.apply(line_length)
            road_gdf[['length_{}'.format(hzd_name),'val_{}'.format(hzd_name)]] = inb[['length_{}'.format(hzd_name),
                                                                                      'val_{}'.format(hzd_name)]] 
        # ADD SOME CHARACTERISTICS OF THE REGION AS COLUMNS TO OUTPUT DATAFRAME
        df = road_gdf.copy()
        df['NUTS-3'] = region
        df['NUTS-2'] = NUTS_down(region)
        df['NUTS-1'] = NUTS_down(NUTS_down(region))
        df['NUTS-0'] = NUTS_down(NUTS_down(NUTS_down(region)))
        
        # ADD THE MISSING LANE DATA
        lane_file = load_config()['filenames']['default_lanes'] #import the pickle containing the default lane data
        with open(os.path.join(input_path,lane_file), 'rb') as handle:
            default_lanes_dict = pickle.load(handle)
        df = df.apply(lambda x: add_default_lanes(x,default_lanes_dict),axis=1).copy() #apply the add_default_lanes function
        
        # LOAD THE DICT REQUIRED FOR CORRECTING THE MAXIMUM DAMAGE BASED ON THE NUMBER OF LANES
        lane_damage_correction = load_lane_damage_correction(map_dam_curves,"Max_damages","G:M") 
        #actual correction is done within the road_loss_estimation function
        
        # PERFORM LOSS CALCULATION FOR ALL ROAD SEGMENTS
        val_cols = [x for x in list(df.columns) if 'val' in x]
        df = df.loc[~(df[val_cols] == 0).all(axis=1)] #Remove all rows from the dataframe containing roads that don't intersect with floods
        
        tqdm.pandas(desc = region)
        
        for curve_name in interpolators:
            interpolator = interpolators[curve_name] #select the right interpolator
            df = df.progress_apply(lambda x: road_loss_estimation(x,interpolator,hzd_names,dict_max_damages,max_damages_HZ,curve_name,
                                                                  lane_damage_correction,log_file=os.path.join(output_path,'road_loss_estimation_log_{}.txt'.format(os.getenv('COMPUTERNAME')))),axis=1)           
        
        # SAVE AS CSV AND AS PICKLE
        df.reset_index(inplace=True,drop=True)
        df.to_csv(os.path.join(output_path ,'{}.csv'.format(region)))
        df.to_pickle(os.path.join(output_path ,'{}.pkl'.format(region)))
        
        if log_file is not None: #write to log file
            file = open(log_file, mode="a")
            file.write("\n\nLoss calculation finished for region: {} at time: {}\n".format(region,time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
            file.close()
        
    except Exception as e:
        print('Failed to finish {} because of {}!'.format(region,e))
        if log_file is not None: #write to log file
            file = open(log_file, mode="a")
            file.write("\n\nFailed to finish {} because of: {}\n".format(region,e))
            file.close()


def default_factory():
    return 'none'

def map_roads(filename,sheet_name):
    """
    Creates a dictionary to create an aggregated list of road types; from an Excel file.
    
    Arguments:
        *filename* (string) - name of the Excel file (should be located in the input_path dir.)
        *sheet_name* (string) - name of the Excel sheetname containing the data
    
    Returns:
        *road_mapping* (Default dictionary) - Default dictionary containing OSM 'highway' variables as keys and the aggregated group names as values
    """
    input_path = load_config()['paths']['input_data'] #folder containing the Excel_file
    mapping = pd.read_excel(os.path.join(input_path,filename),
                        sheet_name=sheet_name,index_col=0,usecols="A:B")
    mapping = mapping.T.to_dict(orient='records')[0]
    road_mapping = defaultdict(default_factory, mapping)
    return road_mapping

def import_damage (file_name,sheet_name,usecols):
    """
    Imports the maximum damage data from an Excel file in the input_path folder
    
    Arguments:
        *file_name* (string) : name of the Excel file (should be located in the input_path folder) 
        *sheet_name* (string) : name of the Excel sheet containing the data
        *usecols* (string) : columns containing the data you want to read, including the column with the road_types e.g. "C:F"
        
    Returns:
        *dict* (Ordered Dictionary) : An ordered dictionary with a group of damage estimates as keys; 
             each value contains another ordered dictionary with as keys the types of roads and as values the damages in Euros
                So you call the output as: dict['Worldbank'] to get a dict with all the damages in WorldBank
                And dict['Worldbank']['motorway'] to get the damage for a motorway according to the worldbank
                
                #From version 0.7 and higher, this structure maybe does not make much sense, because we use upper and lower bounds
    
    """
    input_path = load_config()['paths']['input_data'] #this is where the other inputs (such as damage curves) are located
    df = pd.read_excel(os.path.join(input_path,file_name),
                                     sheet_name=sheet_name,header=[3],usecols=usecols,index_col=0)
    df = df.iloc[df.index.notna(),:] #Drop the empty cells
    odf = OrderedDict() #initialize OrderedDict
    return df.to_dict(into=odf)

def load_lane_damage_correction(filename,sheet_name,usecols):
    """
    Loads the maximum damage correction from an Excel file into an ordered dict.
    
    Argument:
        *filename* (string) - name of the Excel file (should be located in the input_path dir)
        *sheet_name* (string) - name of the excel sheet name
        *usecols* (string) - the columns which have the data (first column should have the road_type keys)
        
    Returns:
        *lane_corr* (OrderedDict) - keys are road_types; values are dicts with key: lane, value = correction factor
            Use like: lane_corr['motorway'][4] -> 1.25 (i.e. correct max damage by +25%)    
    """
    
    input_path = load_config()['paths']['input_data'] #folder containing the Excel_file
    lane_corr_df = pd.read_excel(os.path.join(input_path,filename),
                        sheet_name=sheet_name,header=3,usecols=usecols,index_col=0)
    odf = OrderedDict() #initialize OrderedDict
    lane_corr = lane_corr_df.to_dict(orient='index',into=odf)
    return lane_corr

def apply_lane_damage_correction(lane_damage_correction,road_type,lanes):
    """See load_lane_damage_correction; this function only avoids malbehaviour for weird lane numbers"""
    if lanes < 1: #if smaller than the mapped value -> correct with minimum value
        lanes = 1
    if lanes > 6: #if larger than largest mapped value -> use maximum value (i.e. 6 lanes)
        lanes = 6
    return lane_damage_correction[road_type][lanes]

def load_HZ_max_dam(filename,sheet_name,usecols):
    """
    Loads the maximum damages according to Huizinga from an Excel file
    
    Argument:
        *filename* (string) - name of the Excel file (should be located in the input_path dir)
        *sheet_name* (string) - name of the excel sheet name
        *usecols* (string) - the columns which have the data (first column should have the road_type keys)
        
    Returns:
        *HZ_max_dam* (OrderedDict) - keys are road_types; values are dicts with key: lane, value = correction factor
            Use like: lane_corr['motorway'][4] -> 1.25 (i.e. correct max damage by +25%)    
    """
    
    input_path = load_config()['paths']['input_data'] #folder containing the Excel_file
    lane_corr_df = pd.read_excel(os.path.join(input_path,filename),
                        sheet_name=sheet_name,header=0,usecols=usecols,index_col=0)
    odf = OrderedDict() #initialize OrderedDict
    HZ_max_dam = lane_corr_df.to_dict(orient='index',into=odf)
    return HZ_max_dam


def apply_HZ_max_dam(max_damages_HZ,road_type,lanes):
    """See load_lane_damage_correction; this function only avoids malbehaviour for weird lane numbers"""
    if lanes < 1: #if smaller than the mapped value -> correct with minimum value
        lanes = 1
    if lanes > 6: #if larger than largest mapped value -> use maximum value (i.e. 6 lanes)
        lanes = 6
    return max_damages_HZ[road_type][lanes]

def fetch_roads(osm_data,region, **kwargs):
    """
    Function to extract all roads from OpenStreetMap for the specified region.
        
    Arguments:
        *osm_data* (string) -- string of data path where the OSM extracts (.osm.pbf) are located.

        *region* (string) -- NUTS3 code of region to consider.
        
        *log_file* (string) OPTIONAL -- string of data path where the log details should be written to 
    
    Returns:
        *Geodataframe* -- Geopandas dataframe with all roads in the specified **region**.
        
    """    
    from shapely.wkb import loads
    
    ## LOAD FILE
    osm_path = os.path.join(osm_data,'{}.osm.pbf'.format(region))
    driver=ogr.GetDriverByName('OSM')
    data = driver.Open(osm_path)
          
    ## PERFORM SQL QUERY
    sql_lyr = data.ExecuteSQL("SELECT osm_id,highway,other_tags FROM lines WHERE highway IS NOT NULL")
    
    log_file = kwargs.get('log_file', None) #if no log_file is provided when calling the function, no log will be made
    
    if log_file is not None: #write to log file
        file = open(log_file, mode="a")
        file.write("\n\nRunning fetch_roads for region: {} at time: {}\n".format(region,time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
        file.close()
    
    ## EXTRACT ROADS
    roads=[]
    for feature in sql_lyr: #Loop over all highway features
        if feature.GetField('highway') is not None:
            osm_id = feature.GetField('osm_id')
            shapely_geo = loads(feature.geometry().ExportToWkb()) #changed on 14/10/2019
            if shapely_geo is None:
                continue
            highway=feature.GetField('highway')
            try:
                other_tags = feature.GetField('other_tags')
                dct = OSM_dict_from_other_tags(other_tags) #convert the other_tags string to a dict

                if 'lanes' in dct: #other metadata can be drawn similarly
                    try: 
                        #lanes = int(dct['lanes'])
                        lanes = int(round(float(dct['lanes']),0))
                        #Cannot directly convert a float that is saved as a string to an integer;
                        #therefore: first integer to float; then road float, then float to integer
                    except:
                        if log_file is not None: #write to log file
                            file = open(log_file, mode="a")
                            file.write("\nConverting # lanes to integer did not work for region: {} OSM ID: {} with other tags: {}".format(region,osm_id, other_tags))
                            file.close()
                        lanes = np.NaN #added on 20/11/2019 to fix problem with UKH35
                else:
                    lanes = np.NaN

                if 'bridge' in dct: #other metadata can be drawn similarly
                    bridge = dct['bridge']
                else:
                    bridge = np.NaN        
                
                if 'lit' in dct:
                    lit = dct['lit']
                else:
                    lit = np.NaN
                
            except Exception as e:
                if log_file is not None: #write to log file
                    file = open(log_file, mode="a")
                    file.write("\nException occured when reading metadata from 'other_tags', region: {}  OSM ID: {}, Exception = {}\n".format(region,osm_id,e))
                    file.close()
                lanes = np.NaN
                bridge = np.NaN
                lit = np.NaN
            
            #roads.append([osm_id,highway,shapely_geo,lanes,bridge,other_tags]) #include other_tags to see all available metata
            roads.append([osm_id,highway,shapely_geo,lanes,bridge,lit]) #... but better just don't: it could give extra errors...
    
    
    ## SAVE TO GEODATAFRAME
    if len(roads) > 0:
        return gpd.GeoDataFrame(roads,columns=['osm_id','infra_type','geometry','lanes','bridge','lit'],crs={'init': 'epsg:4326'})
    else:
        print('No roads in {}'.format(region))
        if log_file is not None:
            file = open(log_file, mode="a")
            file.write('No roads in {}'.format(region))
            file.close()

def cleanup_fetch_roads(roads_input, region):
    """
    Makes the road network exactly fit within the region (basically correcting shortcomings in the OSM-extract function):
     1. Removes all the roads that are completely outside the region
     2. For roads intersecting the region border: removes the part of the road outside the region
     
     Arguments:
         *roads_input* (DataFrame) : the road network obtained with the fetch_roads function (expects WGS84)
         *region* (string) : the NUTS-3 region name for which the clipping should be done
         
     Returns:
         *roads_output* (DataFrame) : the clipped and cutted road network
    """ 
    input_path = load_config()['paths']['input_data'] #this is where the other inputs (such as damage curves) are located
    filename = load_config()['filenames']['NUTS3-shape']
    NUTS_poly = gpd.read_file(os.path.join(input_path,filename))
    region_shape = NUTS_poly.loc[NUTS_poly["NUTS_ID"] == region].to_crs({'init':'epsg:4326'}) #import and convert to WGS84
    region_shape_geom = region_shape.iloc[0].geometry #only keep the geometry
    
    #Carry out step 1
    roads_output = roads_input.loc[roads_input['geometry'].apply(lambda x: x.intersects(region_shape_geom))].reset_index(drop=True)
    #Carry out step 2
    roads_output['geometry'] = roads_output.geometry.apply(lambda x: x.intersection(region_shape_geom))
    
    return roads_output  

def OSM_dict_from_other_tags(other_tags):
    """
    Creates a dict from the other_tags string of an OSM road segment
    
    Arguments:
    *other_tags* (string) : string containing all the other_tags data from the OSM road segment
    
    Returns:
    *lanes* (int) : integer containing the number of lines of the road segment
    """
    
    dct = {}
    if other_tags is not None:
        try:
            lst = other_tags.split("\",\"")
            for i in lst:
                j = i.split('=>')
                dct['{}'.format(j[0].replace("\"",""))] =j[1].replace("\"","")
        except:
            print("Dict construction did not work for: {}".format(other_tags))
    return dct
        
def create_hzd_df(geometry,hzd_list,hzd_names):
    """
    Arguments:
        
        *geometry* (Shapely Polygon) -- shapely geometry of the region for which we do the calculation.

        *hzd_list* (list) -- list of file paths to the hazard files.

        *hzd_names* (list) -- list of names to the hazard files.
    
    Returns:
        *Geodataframe* -- GeoDataFrame where each row is a unique flood shape in the specified **region**.
    
    """
    
    ## MAKE GEOJSON GEOMETRY OF SHAPELY GEOMETRY FOR RASTERIO CLIP
    geoms = [mapping(geometry)]

    all_hzds = []

    ## LOOP OVER ALL HAZARD FILES TO CREATE VECTOR FILES
    for iter_,hzd_path in enumerate(hzd_list):
        # extract the raster values values within the polygon 
        with rasterio.open(hzd_path) as src:
            out_image, out_transform = mask(src, geoms, crop=True)

            # change into centimeters and make any weird negative numbers -1 (will result in less polygons)
            out_image[out_image <= 0] = -1
            out_image = np.array(out_image*100,dtype='int32')

            # vectorize geotiff
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                shapes(out_image[0,:,:], mask=None, transform=out_transform)))

            # save to geodataframe, this can take quite long if you have a big area
            gdf = gpd.GeoDataFrame.from_features(list(results))
            
            # this is specific to this calculation: change to epsg:3035 to make sure intersect works.
            gdf.crs = {'init' :'epsg:3035'}
            gdf = gdf.loc[gdf.raster_val >= 0]
            gdf = gdf.loc[gdf.raster_val < 5000] #remove outliers with extreme flood depths (i.e. >50 m)
            gdf['geometry'] = gdf.buffer(0)

            gdf['hazard'] = hzd_names[iter_]
            all_hzds.append(gdf)
    return pd.concat(all_hzds)

def intersect_hazard(x,hzd_reg_sindex,hzd_region):
    """
    Arguments:
        
        *x* (road segment) -- a row from the region GeoDataFrame with all road segments. 

        *hzd_reg_sindex* (Spatial Index) -- spatial index of hazard GeoDataFrame

        *hzd_region* (GeoDataFrame) -- hazard GeoDataFrame
    
    Returns:
        *geometry*,*depth* -- shapely LineString of flooded road segment and the average depth
      
    """    
    matches = hzd_region.iloc[list(hzd_reg_sindex.intersection(x.geometry.bounds))].reset_index(drop=True)
    try:
        if len(matches) == 0:
            return x.geometry,0
        else:
            append_hits = []
            for match in matches.itertuples():
                inter = x.geometry.intersection(match.geometry)
                if inter.is_empty == True:
                    continue
                else:
                    if inter.geom_type == 'MultiLineString':
                        for interin in inter:
                            append_hits.append((interin,match.raster_val))
                    else:
                         append_hits.append((inter,match.raster_val))


            if len(append_hits) == 0:
                return x.geometry,0
            elif len(append_hits) == 1:
                return append_hits[0][0],int(append_hits[0][1])
            else:
                return shapely.geometry.MultiLineString([x[0] for x in append_hits]),int(np.mean([x[1] for x in append_hits]))
    except:
        return x.geometry,0


def sum_tuples(l):
    return tuple(sum(x) for x in zip(*l))

#def loss_estimations_flooding(x,global_costs,paved_ratios,flood_curve_paved,flood_curve_unpaved,events,wbreg_lookup,param_values,val_cols):
#FUNCTION REMOVED IN VERSION 0.3 AND HIGHER; REPLACED BY IMPORT_FLOOD CRUVES AND ROAD_LOSS_ESTIMATION

#FIRST PART OF THE REPLACEMENT OF loss_estimations_flooding
def import_flood_curves(filename,sheet_name,usecols):
    """
    Imports the flood curves from a predefined path
    
    Arguments: 
        *filename* (string) : name of the Excel file (should be located in the input_path folder) e.g. "Costs_curves_Europe.xlsx"
        *sheet_name* (string) : name of the Excel sheet containing the damage curves (e.g. 'curves')
        *usecols* (string) : string with the columns of the Excel sheet you want to import, e.g. "B:AA"
            
    Returns:
        *OrderedDict* : keys are the names of the damage curves
                        values are scipy interpolators
    """
    from scipy.interpolate import interp1d  #import Scipy interpolator function
    from collections import OrderedDict  #Use an ordered dict so that the damage curves will remain in the order of the Excel sheet
    
    input_path = load_config()['paths']['input_data'] #this is where the other inputs (such as damage curves) are located
    flood_curves = pd.read_excel(os.path.join(input_path,filename),
                                 sheet_name=sheet_name,header=[2],index_col=None,usecols=usecols) #removed skip-footer; gave unexpected results
    headers = flood_curves.columns
    curve_name = [0] * int(len(headers)/2) #create empty arrays
    interpolators = [0] * int(len(headers)/2)
    for i in range(0,int(len(headers)/2)):  #iterate over the damage curves in the Excel file
        curve_name[i] = headers[i*2]        
        curve = flood_curves.iloc[:,2*i:2*i+2].dropna()
        #curve x-values in the even; and y-values in the uneven columns
        interpolators[i] = interp1d(curve.values[1:,0], curve.values[1:,1], 
                                    fill_value=(curve.values[1,1],curve.values[-1,1]), bounds_error=False)
    return OrderedDict(zip(curve_name,interpolators)) 


#SECOND PART OF THE REPLACEMENT OF loss_estimations_flooding

def road_loss_estimation(x,interpolator,events,max_damages,max_damages_HZ,curve_name,lane_damage_correction,**kwargs):
    """
    Carries out the damage estimation for a road segment using various damage curves
    
    Arguments:
        *x* (Geopandas Series) -- a row from the region GeoDataFrame with all road segments
        *interpolator* (SciPy interpolator object) -- the interpolator function that belongs to the damage curve
        *events* (List of strings) -- containing the names of the events: e.g. [rp10,...,rp500] 
            scripts expects that x has the columns length_{events} and val_{events} which it needs to do the computation
        *max_damages* (dictionary) -- dictionary containing the max_damages per road-type; not yet corrected for the number of lanes
        *max_damages_HZ* (dictionary) -- dictionary containing the max_damages per road-type and number of lanes, for the Huizinga damage curves specifically
        *name_interpolator* (string) -- name of the max_damage dictionary; to save as column names in the output pandas DataFrame -> becomes the name of the interpolator = damage curve
        *lane_damage_correction (OrderedDict) -- the max_dam correction factors (see load_lane_damage_correction)
        
    Returns:
        *x* (GeoPandas Series) -- the input row, but with new elements: the waterdepths and inundated lengths per RP, and associated damages for different damage curves
    
    """
    try:
        #GET THE EVENT-INDEPENDENT METADATA FROM X
        road_type = x["road_type"] #get the right road_type to lookup ...
        
        #abort the script for not-matching combinations of road_types and damage curves
        if((road_type in ['motorway','trunk'] and curve_name not in ["C1","C2","C3","C4","HZ"]) or
           (road_type not in ['motorway','trunk'] and curve_name not in ["C5","C6","HZ"])): #if combination is not applicable
            for event in events: #generate (0,0,0,0,0) output for each event
                x["dam_{}_{}".format(curve_name,event)]=tuple([0]* 5)
            return x
        
        lanes = x["lanes"] #... and the right number of lanes
        
        #DO THE HUIZINGA COMPARISON CALCULATION
        if curve_name == "HZ": #only for Huizinga
            #load max damages huizinga
            max_damage = apply_HZ_max_dam(max_damages_HZ,road_type,lanes) #dict lookup: [road_type][lanes]
            for event in events:
                depth = x["val_{}".format(event)]
                length = x["length_{}".format(event)] #inundated length in km
                x["dam_{}_{}".format(curve_name,event)]= round(max_damage * interpolator(depth) * length,2)
        
        #DO THE MAIN COMPUTATION FOR ALL THE OTHER CURVES
        else: #all the other curves
            #LOWER AN UPPER DAMAGE ESTIMATE FOR THIS ROAD TYPE BEFORE LANE CORRECTION
            lower = max_damages["Lower"][road_type] #... the corresponding lower max damage estimate ... 
            upper = max_damages["Upper"][road_type] #... and the upper max damage estimate

            #CORRECT THE MAXIMUM DAMAGE BASED ON NUMBER OF LANES
            lower = lower * apply_lane_damage_correction(lane_damage_correction,road_type,lanes)
            upper = upper * apply_lane_damage_correction(lane_damage_correction,road_type,lanes)

            max_damages_interpolated = [lower,(3*lower+upper)/4,(lower+upper)/2,(lower+3*upper)/4,upper] #interpolate between upper and lower: upper, 25%, 50%, 75% and higher
                                                                                                         #if you change this, don't forget to change the length of the exception output as well!         
            for event in events:
                depth = x["val_{}".format(event)] #average water depth in cm
                length = x["length_{}".format(event)] #inundated length in km

                results = [None]* len(max_damages_interpolated) #create empty list, which will later be coverted to a tuple
                for index, key in enumerate(max_damages_interpolated): #loop over all different damage functions; the key are the max_damage percentile
                    results[index] = round(interpolator(depth)*key*length,2) #calculate damage using interpolator and round to eurocents
                x["dam_{}_{}".format(curve_name,event)]=tuple(results) #save results as a new column to series x
    
    #HANDLE EXCEPTIONS BY RETURNING ZERO DAMAGE IN THE APPROPRIATE FORMAT
    except Exception as e:
        errorstring = "Issue with road_loss_estimation, for  x = {} \n exception = {} \n Damages set to zero. \n \n".format(str(x),e)
        log_file = kwargs.get('log_file', None) # get the name of the log file from the keyword arguments
        if log_file is not None: #write to log file
            file = open(log_file, mode="a")
            file.write(errorstring)
            file.close()
        else: #If no log file is provided, print the string instead
            print(errorstring)
            
        for event in events:          
            if curve_name == "HZ":
                x["dam_{}_{}".format(curve_name,event)] = 0
            else:
                x["dam_{}_{}".format(curve_name,event)]=tuple([0]* 5) #save empty tuple (0,0,0,0,0)
                
    return x 

def add_default_lanes(x,default_lanes_dict):
    """
    Add the default number of lanes if the lane data is missing.
    
    Arguments:
    *x* (Geopandas Series) -- a row from the region GeoDataFrame with all road segment; needs to have the columns 'NUTS-0' (the country) and 'road_type'
    *default_lanes_dict (OrderedDict) - keys: NUTS-0 country codes; values: 
        OrderedDicts with keys: road types and values: default number of lanes
                                                        
    Returns:
    *x* with the updated number of lanes
    """
    
    if np.isnan(x.lanes):
        x.lanes = default_lanes_dict[x['NUTS-0']][x['road_type']]
    return x

