"""This script contains function for aggregation of the Lisflood
Damage modelling efforts as well as the OSM-line aggregation approach"""

from collections import OrderedDict
import geopandas as gpd
import numpy as np
import os as os
import pandas as pd
import pickle 
from pdb import * #for debugging
import time as time
import rasterio

from natsort import natsorted
from rasterio.mask import mask
from rasterstats import zonal_stats
from scipy.interpolate import interp1d
from shapely.geometry import mapping
from shapely.geometry import LineString
from tqdm import tqdm
from utils_functions import load_config


"""
GENERAL FUNCTIONS
"""

def AoI_FP(region):
    """
    Determines for each OSM road segment, the flood protection level according to Jongman (AoI functionality commented out)
    
    Arguments:
        *region* (string) -- name of the NUTS-3 region for which to do the analysis
        
    Returns:
        *result_dict* (dictionary) -- key = region; value = keys: osm_id's; values: keys: Jongman FP; AoI per RP.
        
    Known issues: 
        - If raster cells do have inundations but no flood protection level an error occurs, 
          this can be avoid 'nibbling the flood protection Raster, e.g. with QGIS'
    """
    fpl_aoi_output = load_config()['paths']['fpl_aoi_output'] #loaded first, because it is used for exception logging
    try: 
        #load paths
        input_path = load_config()['paths']['input_data']
        output_path = load_config()['paths']['output']
        #AoI_path = load_config()['paths']['aoi_data']
        FP_name = load_config()['paths']['fpl_raster']      
        #load filenames
        NUTS3_shape = load_config()['filenames']['NUTS3-shape']

        #check if path exists
        if os.path.exists(os.path.join(fpl_aoi_output,"{}_AoI_FP.pkl".format(region))): 
            print('{} already finished!'.format(region))
            return None

        df = pd.read_pickle(os.path.join(output_path,"{}.pkl".format(region))) #open the df
        df = df.loc[df['osm_id'] != (0,0,0,0,0)] #remove erratic rows
        df.crs = {'init' :'epsg:4326'}
        df = df.to_crs(epsg=3035) #change to same projection as the raster data
        
        NUTS_regions = gpd.read_file(os.path.join(input_path,NUTS3_shape))
        geometry = NUTS_regions['geometry'].loc[NUTS_regions.NUTS_ID == region].values[0] #The NUTS-3 shape
        geoms = [mapping(geometry)] #Convert to GeoJson file
        
        #AoI_list = natsorted([os.path.join(AoI_path, x) for x in os.listdir(AoI_path) if x.endswith(".tif")]) #examine whole folder, select tifs, sort
        #AoI_names = ['rp10','rp20','rp50','rp100','rp200','rp500']

        tqdm.pandas(desc=region)

        ### ATTRIBUTE THE AOI DATA ###
        #for i, raster_path in enumerate(AoI_list): #iterate over all AoI rasters
        #    df['AoI_{}'.format(AoI_names[i])] = df.geometry.progress_apply(lambda x: zonal_stats(x, raster_path,stats=['majority'],nodata=0,all_touched=True)[0]['majority'])

        ### ATTRIUBTE THE FLOOD PROTECTION DATA ###
        df['Jongman_FP'] = df.geometry.apply(lambda x: zonal_stats(x, FP_name, stats=['majority'],nodata=0,all_touched=True)[0]['majority'])

        #cols = ["osm_id","NUTS-3","Jongman_FP","AoI_rp10","AoI_rp20","AoI_rp50","AoI_rp100","AoI_rp200","AoI_rp500"]
        cols = ["osm_id","NUTS-3","Jongman_FP"]
        df_sel = df[cols]
        df_sel = df_sel.set_index('osm_id')
        df_sel.to_pickle(os.path.join(fpl_aoi_output,"{}_AoI_FP.pkl".format(region)))

        result_dict = {region : df_sel.to_dict(orient='index')}
        print("{} finished!".format(region))
        return(result_dict)
    
    except Exception as e:
        log_mess = 'Failed to finish {} because of {}!'.format(region,e)
        print(log_mess)
        log_file = os.path.join(fpl_aoi_output,'AoI_FP_log_{}.txt'.format(os.getenv('COMPUTERNAME')))
        file = open(log_file, mode="a")
        file.write(log_mess)
        file.close()
        
def EAD_multi(region):
    """
    Calculates the EAD aggregated per road_type using the lighting-mix, saves intermediate results
    
    Arguments:
        *region* (string) - Name of the NUTS-3 region
        
    Returns:
        *results* (DataFrame) - indices are road types; columns the EADS and the name of the region
        
    Also saves the intermediate results as pickles:
        *{NUTS3}_EAD_segment_raw* (DataFrame pickle) - the raw EAD results
        *{NUTS3}_EAD_segment_nomix* (DataFrame pickle) - with GDP correction, bridges dropped (but keeping all the different damage curves)
        *{NUTS3}_EAD_roadtype_nomix* (DataFrame pickle) - idem, but aggregated per road type
        *{NUTS3}_EAD_segment_litmix* (DataFrame pickle) - GDP corrected; bridges dropped; max damage and damage curve according to lighting mix        
        *{NUTS3}_EAD_roadtype_litmix* (DataFrame pickle) - idem, but aggregated per road type
        
    """   
    postproc_output = load_config()['paths']['postproc_output'] 
    out_path = os.path.join(postproc_output,'baseline')
    log_file = os.path.join(out_path,"EAD_multi_{}.txt".format(os.getenv('COMPUTERNAME')))

    try: 
        ### CALCULATE THE EADs FOR THE RAW DATA
        df = EAD_region_segmentwise(region) #returns the dataframe with all individual road segments including the EADs
        df.to_pickle(os.path.join(out_path,"{}_EAD_segment_raw.pkl".format(region)))

        ### DROP ALL THE BRIDGES FROM THE MAIN DATAFRAME
        bridges = df[df['bridge'].notnull()] #new dataframe with all the bridges
        df = df[~df['bridge'].notnull()] #original df with all the bridges dropped

        if not df.empty:                    
            ### CORRECT FOR national GDP ##
            priceyear = 2015
            ratios = GDP_corr_national(priceyear, ref_NUTS0='EU28')
            super_region = df['NUTS-0'].iloc[0]
            factor = ratios[super_region] #the correction factor

            # select the damage/EAD columns
            damcols = [x for x in list(df.columns) if 'dam' in x]
            EADcols = [x for x in list(df.columns) if 'EAD' in x]
            cols = damcols + EADcols #select the damage columns (to increase speed)

            ### DO THE ACTUAL GDP_CORRECTION
            df[cols] = df[cols].applymap(lambda x: smart_multiply(x,factor)) #do the actual computation
            df.to_pickle(os.path.join(out_path,"{}_EAD_segment_nomix.pkl".format(region)))
     
            ### AGGREGATE SELECTED COLUMNS BY ROAD TYPE
            agg_cols = ["EAD_C1","EAD_C2","EAD_C3","EAD_C4","EAD_C5","EAD_C6","EAD_HZ"]
            aggregate_tuplecollection(df,agg_cols,"NUTS-3").to_pickle(os.path.join(out_path,"{}_EAD_roadtype_nomix.pkl".format(region)))
      
            #DO THE LIGHTING MIX on a copy of the df.
            df2 = df.copy()
            df2 = df2.apply(lighting_blend,axis=1)
            df2.to_pickle(os.path.join(out_path,"{}_EAD_segment_litmix.pkl".format(region)))

            ### AGGREGATE THE LITMIX RESULTS FOR SELECTED COLUMNS
            rescols = ['road_type','EAD_HZ','EAD_lowflow','EAD_highflow'] #Change this if you don't do the lighting mix
            summary = df2[rescols].groupby('road_type').sum()
            summary['NUTS-3'] = region
            summary.to_pickle(os.path.join(out_path,"{}_EAD_roadtype_litmix.pkl".format(region)))
   
            ### Write to logfile and print
            message = "\n {}: calculation of EAD successfull".format(region)
        
        else:
            message = "\n {}: DataFrame containing output of main script is empty (at least after removing bridges)".format(region)
        
        file = open(log_file, mode="a")
        file.write(message)
        file.close()
        print(message)
    
    except Exception as e:
        
        ### Write to logfile and print
        message = "\n {}: an error occured: {}".format(region,e)
        file = open(log_file, mode="a")
        file.write(message)
        file.close()
        print(message)
        
        return None

    return None

def EAD_region_segmentwise(region):
    """
    Calculates Expected Annual Damages for one NUTS-3 region, with keeping all the individual road segments
    (For example used to make maps)
    
    Arguments:
        *region* (string) - the NUTS-3 name of the region
    
    Returns:
        *mdf* (Geopandas DataFrame) - Merged DF containing the model results; the flood protection level and AOI; and the EAD calculated using these
        
    Note that when using this function, the GDP-correction still has to be done, as well as bridge filtering and using lighting meta-data.    
    
    """
    #FIND PATHS CONTAINING MODEL RESULTS, FLOOD PROTECTION LEVELS AND AREA OF INFLUENCE
    output_path = load_config()['paths']['output']
    fpl_aoi_output = load_config()['paths']['fpl_aoi_output']
    output_file = os.path.join(output_path,'{}.pkl'.format(region))
    fpl_aoi_file = os.path.join(fpl_aoi_output,'{}_AoI_FP.pkl'.format(region))
    
    #CHECK IF PATHS EXIST:
    if not os.path.exists(output_file):
        raise FileNotFoundError("The file containing the model results: {} does not exist.".format(output_file))
    if not os.path.exists(fpl_aoi_file):
        raise FileNotFoundError("The file containing the flood protection level and area of influence: {} does not exist.".format(fpl_aoi_file))
    
    #READ REGION RESULTS DATAFRAME
    df = pd.read_pickle(output_file)
    df = df.set_index('osm_id') #use the osm_id column as an index
    
    #READ FLOOD PROTECTION AND AREA OF INFLUENCE DATAFRAME
    fpl_aoi = pd.read_pickle(fpl_aoi_file)
    
    #MERGE BOTH DATAFRAMES
    mdf = pd.merge(df,fpl_aoi,how='outer',on='osm_id') #merged dataframe
    mdf.drop(labels='NUTS-3_y',inplace=True,axis=1) #double because of the merge
    mdf.rename({'NUTS-3_x':'NUTS-3'},inplace=True,axis=1) #and change the name back
    
    #DO THE RISK CALCULATION
    RPs = [500,200,100,50,20,10]
    curves = ["C1","C2","C3","C4","C5","C6","HZ"]
    positions = [0,1,2,3,4]
    tqdm.pandas(desc="EAD_{}".format(region))
    mdf = mdf.progress_apply(lambda x: EAD_asset(x,RPs,curves,positions), axis=1)

    return mdf

#CORE MODEL? YES - BUT VERY SPECIFIC FOR THE SET OF DAMAGE CURVES USED IN THE EU-IMPLEMENTATION
def EAD_asset(x,RPs,curves,positions):
    """
    Calculates the expected annual damage for one road segment (i.e. a row of the DataFrame containing the results)
    based on the damage per return period and the flood protection level.
    
    
    WARNING: THIS FUNCTION PROBABLY REALLY SLOWS DOWN THE OVERALL POST-PROCESSING SCRIPT (COMPARE ITS PERFORMANCE WITH
    THE CLIMATE CHANGE SCRIPT) Most likely, the cause is the use of a loop and an if-statement for the manipulation; this is not
    smart for a function that is applied on a DataFrame!!!
    
    Arguments:
        *x* (Geopandas Series) - A row of the GeoPandas DataFrame containing all road segments, should have dam cols, rps, flood protection level 
        *RPs* (List) - return periods of the damage data, in descending order e.g. [500,200,100,50,20,10]
        *curves* (List) - short names of damage curves for which to calculate the EAD e.g. ["C1","C2","C3","C4","C5","C6","HZ"]
        *positions* (List) - tuple positions of available max damage estimates e.g. [0,1,2,3,4]
    
    Returns:
        *x* (Geopandas Series) - with added the new columns containing EADs
    """
    PL = x["Jongman_FP"]
    RPs_copy = [y for y in RPs] #make a new list, or it will be altered in the function and mess up everything!!!
    for curve in curves:
        damcols = ["dam_{}_rp{}".format(curve,rp) for rp in RPs] #this was RPs; but somehow this did not work
        EAD = [0,0,0,0,0] #initialize empty lists
        for pos in positions: #iterate over all the max damage estimates
            dam = list(x[damcols].apply(lambda y: pick_tuple(y,pos)).values) #creates a numpy array with the damage values of the desired max_dam
            EAD[pos] = risk_FP(dam,RPs_copy,PL) 
        if not curve == "HZ": #save results to the series, which will be returned as a row in the df
            x["EAD_{}".format(curve)] = tuple(EAD)
        else: 
            x["EAD_HZ"] = EAD[0]
    return x

#CAN BE REPLACED BY VECTORIZED FUNCTION USED IN THE CLIMATE CHANGE MODULE
def risk_FP(dam,RPs,PL):
    """
    Calculates the flood risk from damage estimates and corresponding return periods by trapezoidal integration, 
    accounting for the flood protection in place. 
    
    Arguments:
        *dam* (list) - damage estimates (e.g. Euro's) - from high to low RPs
        *RPs* (list) - the return periods (in years) of the events corresponding to these damage estimates - from high to low RPs (order of both lists should match!) 
        *PL* (integer) - the flood protection level in years
    
    Returns:
        *risk* (float) - the estimated flood risk in Euro/y
    
    Note that this calculation is not trivial and contains important assumptions about the risk calculation:
     - Damage for RPs higher than the largest RP equals the damage of the largest RP
     - Damage for RPs lower than the smallest RP is 0.
     - Damage of events in between known RPs are interpolated linearly with RP.
    """
    if not sorted(RPs, reverse=True) == RPs:
        raise ValueError('RPs is not provided in the right format. Should be a descending list of RPs, e.g. [500,100,10]')
    
    if RPs[-1] < PL < RPs[0]: #if protection level is somewhere between the minimum and maximum available return period
        pos = RPs.index(next(i for i in RPs if i < PL)) #find position of first RP value < PL; this is the point which need to be altered
        dam = dam[0:pos+1] #remove all the values with smaller RPs than the PL
        dam[pos] = np.interp(x=(1/PL),xp=[(1/RPs[pos-1]),(1/RPs[pos])],fp=[dam[pos-1],dam[pos]]) #interpolate the damage at the RP of the PL
        #note that you should interpolate over the probabilities/frequences (therefore the 1/X), not over the RPs; this gives different results
        RPs[pos] = PL #take the PL as the last RP...
        RPs = RPs[0:pos+1] #... and remove all the other RPs

    elif PL >= RPs[0]: #protection level is larger than the largest simulated event
        return (1/PL) * dam[0] #damage is return frequence of PL times the damage of the most extreme event

    #not necessary to check other condition (PL <= 10 year -> then just integrate over the available damages, don't assume additional damage)
     
    dam.insert(0,dam[0]) #add the maximum damage to the list again (for the 1:inf event)
    Rfs = [1 / RP for RP in RPs] #calculate the return frequencies (probabilities) of the damage estimates
    Rfs.insert(0,0) #add the probability of the 1:inf event
    integral = np.trapz(y=dam,x=Rfs).round(2)
    return integral



#CORE MODEL? NO, SPECIFIC FOR EUROPE IMPLEMENTATION
def lighting_blend(r):
    """
    Create a unique blend of max_damage estimates using the lighting metadata.
    
    Arguments:
        *r* (road segment) -- a row from the merged results dataframe; should have max damages AND EADs 
    
    Returns:
        *r* (road segment) -- the same row, with tuples replaced and add cols 'EAD_lowflow' and 'EAD_highflow'
    
    The tuple damages and EAD's are replaced:
    For motorways; the max_damage value is the 25% or 75%, based on the lighting metadata
    For all other roads, the 50% value is taken
    Scripts assumes all damage inputs as lenght-5 tuple: (0,0,0,0,0)
    
    """
    #select the damage/EAD columns; but not the HZ
    damcols = [x for x in list(r.keys()) if 'dam' in x]
    damcols = [x for x in damcols if 'HZ' not in x] #doesn't make sense to do this for the HZ cols
    EADcols = [x for x in list(r.keys()) if 'EAD' in x]
    EADcols = [x for x in EADcols if 'HZ' not in x] #doesn't make sense to do this for the HZ cols
    cols = damcols + EADcols #select the damage columns (to increase speed)
    
    #MIX THE MOST FITTING MAXIMUM DAMAGE
    if r['road_type'] in ['motorway','trunk']:
        if r['lit'] in ['yes','24/7','automatic','disused']: #missing operating times now, 
                    #but this is very unfrequently used ('no' + 'yes' already cover >98% of tag occurence)
            pos = 3 #select the 3rd max damage estimate
        else:
            pos = 1 #
    else:
        pos = 2
        
    for col in cols: #replace the tuple with the requested 
        r[col] = r[col][pos]
    
    #MIX THE MOST APPROPRIATE DAMAGE CURVES
    if r['road_type'] in ['motorway','trunk']:
        if r['lit'] in ['yes','24/7','automatic','disused']: #missing operating times now
            r['EAD_lowflow'] = r['EAD_C1'] #select the expensive (accessories) road
            r['EAD_highflow'] = r['EAD_C2']
        else:
            r['EAD_lowflow'] = r['EAD_C3'] #select the cheap (no accessories) road
            r['EAD_highflow'] = r['EAD_C4']
    else:
            r['EAD_lowflow'] = r['EAD_C5']
            r['EAD_highflow'] = r['EAD_C6']
    
    return r

#CORE MODEL? NO - ONLY FOR EUROPE IMPLEMENTATION
def lighting_blend_only_curves(r):
    """
    Script to further process the lighting mix baseline postprocessing outputs.
    Script takes the average of the high-flow and low-flow estimates for all road_types.
    For motorways/trunks also makes a choice between the sophisticated and simple curves, 
        using the same rules as the function "lighting_blend "
    
    
    Arguments:
        *r* (Geopanda Series) -- a row from the merged results dataframe; should have columns with damages per damage curves
    
    Returns:
        *r* (Geopandas Series) -- the same row, with a set of new columns dam_lit_rpXX
    
    The tuple damages and EAD's are replaced:
    For motorways; the max_damage value is the 25% or 75%, based on the lighting metadata
    For all other roads, the 50% value is taken
    Scripts assumes all damage inputs as lenght-5 tuple: (0,0,0,0,0)
    
    Maybe this script can be removed if the above function is designed a little bit smarter
    
    """
    #select the damage/EAD columns; but not the HZ
    damcols = [x for x in list(r.index()) if 'dam' in x]
    damcols = [x for x in damcols if 'HZ' not in x] #doesn't make sense to do this for the HZ cols
    #EADcols = [x for x in list(r.keys()) if 'EAD' in x]
    #EADcols = [x for x in EADcols if 'HZ' not in x] #doesn't make sense to do this for the HZ cols
    cols = damcols# + EADcols #select the damage columns (to increase speed)
    
    #MIX THE MOST FITTING MAXIMUM DAMAGE
    if r['road_type'] in ['motorway','trunk']:
        if r['lit'] in ['yes','24/7','automatic','disused']: #expensive road types
            for rp in [10,20,50,100,200,500]: 
                r['dam_lit_rp{}'.format(rp)] = r[['dam_C1_rp{}'.format(rp),'dam_C2_rp{}'.format(rp)]].mean() 
        else:
            for rp in [10,20,50,100,200,500]: #simple road types
                r['dam_lit_rp{}'.format(rp)] = r[['dam_C3_rp{}'.format(rp),'dam_C4_rp{}'.format(rp)]].mean() 
    else: #all other road types
        for rp in [10,20,50,100,200,500]:
            r['dam_lit_rp{}'.format(rp)] = r[['dam_C5_rp{}'.format(rp),'dam_C6_rp{}'.format(rp)]].mean() 
    
    return r



def create_gridlines(ps,ms,point_spacing=1000):
    """
    Create GeoSeries containing parallels in WGS84 projection.
    
    Arguments:
        *ps* (list) - Parallel coordinates (degrees) of the lines to plot (e.g. [40,50,60,70])
        *ms* (list) - Meridian coordinates (degrees) of the lines to plot (e.g. [-30,-20,-10,0,10,20,30,40,50,60,70])
        *pointspacing* (integer) - Number of points to create (to draw smooth line) (e.g. 100)
    
    Returns:
        *P_series,M_series* (Geopandas GeoSeries) - Contains the parallels and meridians
    """

    #create parallels
    Parallels = []
    start = ms[0]
    end = ms[-1]
    x_values = np.linspace(start,end,point_spacing)

    for p in ps:
        Points = []
        for x in x_values:
            point = (x,p)
            Points.append(point)
        Parallel = LineString(Points)
        Parallels.append(Parallel)

    P_series = gpd.GeoSeries(Parallels,crs='EPSG:4326')
    

    #create meridians
    Meridians = []
    start = ps[0]
    end = ps[-1]
    y_values = np.linspace(start,end,point_spacing)

    for m in ms:
        Points = []
        for y in y_values:
            point = (m,y)
            Points.append(point)
        Meridian = LineString(Points)
        Meridians.append(Meridian)

    M_series = gpd.GeoSeries(Meridians,crs='EPSG:4326')
    
    return P_series,M_series
        

"""
FUNCTIONS FOR AGGREGATION OF LINE-BASED OSM DAMAGE APPROACH
"""

def sum_tuples(l):
    return tuple(sum(x) for x in zip(*l))

def aggregate_tuplecollection(df,dam_cols,results_row_name):
    """
    Aggregate tuple-wise (...,...,...) stored data in a DataFrame,
    while grouping by road_type <todo: generalize this>
    If a column contains floats rather then tuples, it will do a normal aggregation
    
    Arguments: 
       *df* (Pandas dataframe) -- each row is OSM road segment, which at least has the columns: 
               "road_type": containing the values used for the grouping (e.g. motorway, trunk etc.)
               at least one damage column
               a column used for naming the aggregated results
        *dam_cols* -- names of the columns containing the values (tuples or floats) to aggregate
        *result_row_name* -- metadata column which will be added to the result, usually NUTS-3
        
    
    Returns:
        *output* (Pandas DataFrame):
              index = "road_type"
              aggregated columns
              result_row_name column       
    
    """
    sum_as_float = []
    sum_as_tuple = []
    for dam_col in dam_cols:
        if isinstance(df[dam_col][0],np.float): #distinguish between float and tuples in the input df.
            sum_as_float.append(dam_col)
        else:
            sum_as_tuple.append(dam_col)
            
    types = df.groupby('road_type')
    output = types[sum_as_tuple].agg(sum_tuples) #sum the tuples tuple-wise
    output[sum_as_float] = types[sum_as_float].agg('sum') #and the floats by a simple column summation
    output[results_row_name] = df[results_row_name][0] #TODO: also enable the option to give a string as a name
    return output

"""Script to prepare the aggregated results """
def select_tup(tup,position):
    """
    Input the tuple that should go in
    And the position of the value in the tuple you want to come out
    """
    return tup[position]

def pick_tuple(var,pos):
    "If a tuple, returns the value in position pos; else return the input variable"
    if isinstance(var, tuple):
        return var[pos]
    else:
        return var

def df_select_tup(df,position):
    """In the whole dataframe, abstract one value from the tuple and return as a new df"""
    rps = ['dam_rp10','dam_rp20','dam_rp50','dam_rp100','dam_rp200','dam_rp500']
    df_sel = pd.DataFrame(columns=df.columns)
    df_sel.region=df.region
    for rp in rps:
        df_sel[rp] = df[rp].apply(select_tup, position=position)
    return df_sel

def df_select_tup2(df,position,cols):
    """In the whole dataframe, abstract one value from the tuple and return as a new df"""
    df_sel = pd.DataFrame(columns=df.columns)
    df_sel.region=df.region
    for rp in cols:
        df_sel[rp] = df[rp].apply(select_tup, position=position)
    return df_sel

def smart_multiply(var, factor):
    """Elementwise multiplication if tuple, normal if float or integer; else: do nothing. Can be used in a apply_map on a whole dataframe."""
    if isinstance(var,tuple):
        var = tuple(factor*x for x in var)
    elif isinstance(var,float):
        var = factor * var
    elif isinstance(var,int):
        var = factor * var
    return var

# eg. df_select_tup(OSM_outcomes,2).head() #returns a dataframe with all the regions in it, but for only one of the damage curves.

def df_OSM_transp(OSM_outcomes,curve,rp,regions):
    """From the OSM results dataframe, prepare a new dataframe ready for plotting:
    Input: OSM_outcomes dataframe
            curves
            requested return period
            requested regions
    """
    df = df_select_tup(OSM_outcomes,curve) #select the right curve
    df_new = pd.DataFrame(columns=df.loc[df['region'] == regions[0]].index) 
    
    for region in regions:
        df_sel = df.loc[df['region'] == region] #select the right region
        df_sel = df_sel[rp].rename(region)
        df_new = df_new.append(df_sel)

    return df_new

def df_OSM_transp2(OSM_outcomes,curve,max_dam,rp,regions):
    """From the OSM results dataframe, prepare a new dataframe ready for plotting:
    
    Arguments:
        *OSM_outcomes* (Pandas DataFrame): 
            indices are road types, 
            columns represent damages in the format DamageSeries_dam_rpXX,
            last column contains the corresponding NUTS3-region
            values are tuples containing the damage for different damage curves
        
        *curve* (integer) : integer position indicating the damage curve to be used
        *max_dam* (string) : name of the damage series to return
        *rp* (int) : return period to return
        *regions* (list of strings) : containing names of NUTS3-regions to plot
    """
    cols = ["{}_dam_rp{}".format(max_dam,i) for i in [10,20,50,100,200,500] ] #this should not be hard-coded
    df = df_select_tup2(OSM_outcomes,curve,cols) #select the right curve
    df_new = pd.DataFrame(columns=df.loc[df['region'] == regions[0]].index) 
    
    for region in regions:
        df_sel = df.loc[df['region'] == region] #select the right region
        df_sel = df_sel["{}_dam_rp{}".format(max_dam,rp)].rename(region)
        df_new = df_new.append(df_sel)
    
    return df_new

#CORE MODEL? NO but useful for EU-projects
def NUTS_pkl(): #creates a pickle with the NUTS-3 regions as a list
    from main_functions import load_config
    NUTS_3 = gpd.read_file('D:\\ginkel\\Europe_trade_disruptions\\NUTS-2_shapes\\NUTS_2016\\NUTS_RG_01M_2016_3035_LEVL_3.shp')
    N3s = list(NUTS_3['NUTS_ID'].values)
    input_path = load_config()['paths']['input_data']
    pickle.dump(N3s, open(os.path.join(input_path,"NUTS3-names_new.pkl"), "wb" ) )

#CORE MODEL? NO but useful for EU-projects
def NUTS_up(NUTS,to3):    
    """
    For a given NUTS-region, give a list of the names of the underlying NUTS-regions
    
    Arguments:
        *NUTS* (string) -- name of the NUTS region (e.g NL41 )
        *to3* (boolean) -- boolean indicating if the process should be continued up till level-3 is reached. If false, aggregation is only done 1 level up (eg. 0->1; or 1->2)
    
    Returns:
        *out* (list of strings) -- names of the NUTS regions of the higher level (eg. ['NL411','NL412' etc.])
    """
    from main_functions import load_config
    input_path = load_config()['paths']['input_data']
    N3s = pickle.load(open(os.path.join(input_path,"NUTS3-names.pkl"),"rb"))
    d32 = dict.fromkeys(set([str(i)[:-1] for i in N3s]), 0)
    for key in d32:
        d32[key] = [i for i in N3s if str(i)[:-1] == key]
    d21 = dict.fromkeys(set([str(i)[:-1] for i in list(d32.keys())]), 0)
    for key in d21:
        d21[key] = [i for i in list(d32.keys()) if str(i)[:-1] == key]
    d10 = dict.fromkeys(set([str(i)[:-1] for i in list(d21.keys())]), 0)
    for key in d10:
        d10[key] = [i for i in list(d21.keys()) if str(i)[:-1] == key]

    if len(NUTS) >= 6:
        out = None
        raise ValueError("Cannot NUTS_up this value: {}, maybe you want to use NUTS_down instead?".format(NUTS))
    elif len(NUTS) == 4: #looks like a nuts-2 code
        out = d32[NUTS]
    elif len(NUTS) == 3: #looks like a nuts-1 code
        out = d21[NUTS]
        if to3: #one extra step of aggregation
            res = list()
            for b in out:
                res = res + d32[b]
            out = res
    elif len(NUTS) == 2: #looks like a nuts-0 (country) code
        out = d10[NUTS] 
        if to3: #two extra steps of aggregation
            res = list()
            for b in out:
                res = res + d21[b]
                res2 = list()
                for c in res:
                    res2 = res2 + d32[c]
                out = res2
    else:
        out = None
        raise ValueError("Cannot NUTS_up this value: {}".format(NUTS))
    return out

#CORE MODEL? NO but useful for EU-projects
def NUTS_down(NUTS):
    """
    For a given NUTS-region, finds the corresponding region its is part of 
    
    Arguments:
        *NUTS* (string) -- name of the NUTS region (e.g NL413 )
    
    Returns:
        *NUTS_lower* (string) -- name of the NUTS region one level lower (e.g. NL41)
    """
    if len(NUTS) <= 2: #the NUTS-0 level (country) is the lowest level and has length 2
        raise ValueError("Cannot aggregate {} to a lower NUTS-level".format(NUTS))
    #would be nice to check if the output exists as a region
    return str(NUTS)[:-1]

def NUTS_0_list(**kwargs):
    """
    Returns a list with NUTS-0 codes of European countries
    
    Optional keyword arguments:
            "EU28"        : returns EU28 - default True
            "EFTA"        : includes 4 EFTA countries (Iceland, Liechtenstein, Norway, Switzerland) - default False 
            "CAND"        : includes candidate member states - default False
            
    #Prefered workflow:
     - call NUTS_0_list
     - manually filter out countries that you want to exclude
     - NUTS-up from NUTS0 to NUT3 with function NUTS_up()
     - automatically filter out remote NUTS-3 regions with NUTS_3_remote()
    """
    options = {'EU28' : True, 'EFTA' : False, 'CAND' : False} #default keyword arguments
    options.update(kwargs)
    
    l =[]
    
    if options['EU28']: #THE EU-28 COUNTRIES
        l.extend(["AT", "BE", "BG", "CY", "CZ", "DE", 
             "DK", "EE", "EL", "ES", "FI", "FR", 
             "HR", "HU", "IE", "IT", "LT", "LU", 
             "LV", "MT", "NL", "PL", "PT", "RO", 
             "SE", "SI",  "SK", "UK"])
    
    if options['EFTA']: #ADD THE EFTA COUNTRIES
        l.extend(['CH','IS','LI','NO'])

    if options['CAND']: #ADD THE CANDIDATE COUNTRIES
        l.extend(['AL','ME','MK','RS','TR'])

    l.sort()
    
    return l 

def EU28_3l():
    """
    Returns a list with the 3-letter codes of EU-28 members
    Names according to ISO 3166
    
    These are for example used in the IIASA SSP database
    """

    EU28_3l = [ #three-letter codes ISO 3166
          "AUT", # Austria
          "BEL", # Belgium
          "BGR", # Bulgaria
          "CYP", # Cyprus
          "CZE", # Czech Republic
          "DEU", # Germany
          "DNK", # Denmark
          "EST", # Estonia
          "GRC", # Greece
          "ESP", # Spain
          "FIN", # Finland
          "FRA", # France
          "HRV", # Croatia
          "HUN", # Hungary
          "IRL", # Ireland
          "ITA", # Italy
          "LTU", # Lithuania
          "LUX", # Luxembourg
          "LVA", # Latvia
          "MLT", # Malta
          "NLD", # Netherlands
          "POL", # Poland
          "PRT", # Portugal
          "ROU", # Romania
          "SWE", # Sweden
          "SVN", # Slovenia
          "SVK", # Slovakia
          "GBR"  # United Kingdom
         ]
    return EU28_3l

#def NUTS_3_list():
#    """
#    DEPRECIATED
#    Returns a list with all regions with NUTS-3 codes
#    """
#    input_path = load_config()['paths']['input_data']
#    N3s = pickle.load(open(os.path.join(input_path,"NUTS3-names.pkl"),"rb"))
#    return N3s

### TODO: merge some of the above regions ###

def NUTS_3_remote(**kwargs):
    """
    Returns a list with remote NUTS-3 regions you probably don't want to plot
    
    Optional keyword arguments (boolean):
            "Overseas"    : Removes remote, overseas areas (default True)
            "Creta"       : Greek island Creta (default False)
            "Spain"       : Ceauto and Melilla (Spanish North Coast) (default True)
            
    #Suggested syntax for filtering a list: 
    [e for e in LIST if e not in NUTS_3_remote()]
    """
    options = {'Overseas' : True, 'Creta' : False, 'Spain' : True} #default keyword arguments
    options.update(kwargs)
    
    l =[]
    
    if options['Overseas']: #Remote, overseas areas
        l.extend(['PT200','PT300', #Portugal: Azores and Madeira
             'ES703','ES704','ES705','ES706','ES707','ES708','ES709', #Spain: Canary Islands
             'FRY10','FRY20','FRY30','FRY40','FRY50']) #France: overseas areas: Gouadeloupe, Martinique, French Guiana, La Reunion, Mayotte "])
    
    if options['Creta']:
        l.extend(['EL431','EL432','EL433','EL444'])

    if options['Spain']: #Ceauto and Melilla: autonomous Spanish city at the North Coast of Africa
        l.extend(['ES630','ES640'])

    l.sort()
    
    return l 

def GDP_corr_national(priceyear, **kwargs):
    """
    Creates national GDP correction factors (compared to ref code) for a given priceyear
    
    Arguments:
        *priceyear* (int) - the year from which the GDP data should be taken; complete for 2002-2016
        *ref_NUTS0* (str) [optional] - the ref region for which to correct, default = "EU28"
        
    Returns:
        *ratios* (OrderedDict) - key: NUTS-0 codes; values: the correction 
    """
    #from main_functions import load_config
    ref_NUTS0 = kwargs.get('ref_NUTS0', "EU28")
    
    input_path = load_config()['paths']['input_data']
    filename = "Eurostat_realGDPpercap_NUTS0.xls"
    GDP_data = pd.read_excel(os.path.join(input_path,filename),
                        sheet_name="Sheet0",index_col=0,header=2,usecols="A:AL",skipfooter=9)
    GDP_data.drop(columns=[s for s in GDP_data.columns.values if 'Unnamed' in s], inplace=True) #drop the columns containing footnotes
    GDP_data.index.names = ['NUTS_0'] #change the name of the index col

    GDP_ref = GDP_data.loc[ref_NUTS0,str(priceyear)] #lookup the reference priceyear
    ratios = OrderedDict() #create empty ordered dict
    for index, row in GDP_data.iterrows():
        ratios[index] = row[str(priceyear)]/GDP_ref
        
    #ASSUMPTIONS FOR MISSING REGIONS (LIECHTENSTEIN, MONTENEGRO,TURKEY)
    ratios["LI"] = ratios["LU"] #take the highest value (of Luxembourgh)
    ratios["ME"] = ratios["RS"] #comparable to Serbia
    ratios["TR"] = ratios["RO"] #comparable to Romania 
    return ratios


def aggregate_OSM_regions_pkl(regions,OSM_result_dir):
    """
    Aggregates the OSM-results for a number of regions and returns as a DataFrame

    Arguments:
        *regions* (list of strings) -- List containing the names of the NUTS-3 regions over which to aggregate
        *OSM_result_dir* (string) -- Directory containing the model results (should have .pkl files)

    Returns:
        *OSM_outcomes* (Pandas DataFrame) -- rows: combinations of road_types (index, e.g. 'motorway') and NUTS-3 regions (col 1; e.g. "DEB21"); columns: damage columns for different damage curves and return periods; contain tuples or floats with results
        *missing_files* (list) -- names of the regions for which no file was found
        *exceptions* (list of string) -- exceptions that occured during the aggregation

    """
    #check if all the required files are available
    missing_files = [] #array to save the missing regions
    exceptions = []
    file_paths = [] #list to save the usuable filepaths
    for region in regions:
        if not os.path.exists(os.path.join(OSM_result_dir,"{}.pkl".format(region))):
            missing_files.append(region) #add to missing region list
        else:
            file_paths.append(os.path.join(OSM_result_dir,"{}.pkl".format(region))) #append to the list of csv to be opened

    #ini = pd.read_pickle(file_paths[0]) #open the first pkl to initialize the loop -> this gives a problem for Denmark (where the first CSV to open was empty)
    ini = pd.read_pickle(os.path.join(OSM_result_dir,"NL33B.pkl")) #use this region in the Netherlands instead, it seems to have all the data required
    dam_cols = [s for s in ini.columns.values if 'dam' in s] #select the columns containing damage estimates    
    cols = ['NUTS-3'] + dam_cols 
    OSM_outcomes = pd.DataFrame(columns=cols) #create empty dataframe

    #Do the actual work
    for path in tqdm(file_paths):
        try:
            data = pd.read_pickle(path)
            data_agg = aggregate_tuplecollection(data,dam_cols,"NUTS-3")
            OSM_outcomes=pd.concat([OSM_outcomes,data_agg],sort=False)
        except Exception as e:
            exceptions.append("This did not work for: {}, because of {}".format(path,e)) #most of them seem to be just empty csv files
    
    OSM_outcomes.index.name = 'road_type'
    
    return OSM_outcomes, missing_files, exceptions

### FUNCTIONS TO ENABLE THE SAMPLING FROM THE EAD-SEGMENT_NOMIX.pkl files ###
def sample_OSM_regions_pkl(region):
    """
    From the deterministic EAD_segment_nomix.pkl for a region, 
    this script samples 100 possible probabilistic realisation
    
    Arguments:
        *region* (string) : Name of the NUTS-3 region
        This is used to open the pickle in (postproc_output_dir)/(region)_EAD_segment_nomix.pkl
    
    Returns:
        *samples* (Panda Series) : 
    """
    print("{} sample OSM regions started".format(region))
    
    #Suppress a Pandas Warning
    pd.set_option('mode.chained_assignment', None)
    
    postproc_output = load_config()['paths']['postproc_output'] 
    data_path = os.path.join(postproc_output,'baseline',"{}_EAD_segment_nomix.pkl".format(region))
    df = pd.read_pickle(data_path) #open the dataframe containing the EAD_segment_nomix results
    
    #SELECT THE RELEVANT DATA FROM THIS DATAFRAME
    dfA = df[['road_type','lit', 'EAD_C1', 'EAD_C2', 'EAD_C3', 'EAD_C4', 'EAD_C5', 'EAD_C6']]
    
    #SPLIT THE DATASET IN TWO GROUPS
    df1 = dfA[dfA['road_type'].isin(['motorway','trunk'])]
    df2 = dfA[dfA['road_type'].isin(['primary','secondary','tertiary','other'])]
    
    #PROCESS THE ROADS WITH STREET LIGHTING (SOPHISTICATED)
    df11 = df1[~df1.lit.isna()] #no No street lighting (mostly with street lighting 'yes or '24/7')
    df111 = df11[['EAD_C1','EAD_C2']] #Choose relevant damage curve
    df111.rename(columns={'EAD_C1':'low','EAD_C2':'high'},inplace=True) #Rename relevant damage curve
    df112 = df111.apply(lambda x: pick_tuples_randint(x,2,5),axis=1) #Choose on of the last three values as max damage
    df112
    
    #PROCESS ROADS WITHOUT STREET LIGHTING (CHEAP)
    df12 = df1[df1.lit.isna()] #no street lighting
    df121 = df12[['EAD_C3','EAD_C4']] #Choose relevant damage curve
    df121.rename(columns={'EAD_C3':'low','EAD_C4':'high'},inplace=True) #Rename relevant damage curve
    df122 = df121.apply(lambda x: pick_tuples_randint(x,0,3),axis=1) #Choose one of the first 3 values as max damage
    df122
    
    #PROCESS THE OTHER ROADS
    df21 = df2[['EAD_C5','EAD_C6']]
    df21.rename(columns={'EAD_C5':'low','EAD_C6':'high'},inplace=True)
    df22 = df21.apply(lambda x: pick_tuples_randint(x,0,5),axis=1) #Randomly choose any of the tuples 
    df22
    
    #DRAW VALUES FROM THE NORMAL DISTRIBUTION TO ENABLE THE SAMPLING
    zs = zs_random_1000() #always draw the same 100 or 1000 z-scores
    sdvs = 2 #assume the amount of standard deviations to which the known min and max flow velocity equal
    
    #REMERGE ALL THE DATAFRAMES
    dfT = pd.concat([df112,df122,df22])
    dfT.name = region
    dfT2 = dfT.apply(sample_zees,result_type='expand',axis=1,zs=zs)
    dfT2[dfT2 <= 0] = 0 #remove all damages < 0 euro (truncate the normal distribution)
    dfT3 = dfT2.sum(axis=0)
    dfT4 = dfT3.drop(labels=['low','high'])
    dfT4.name = region
    dfT4.to_pickle(os.path.join(postproc_output,'baseline',"{}_EAD_total_sampled_1000.pkl".format(region)))
    print("{} sample OSM regions finished".format(region))
    return dfT4

def sample_zees(series,zs,sdvs=2):
    """
    Interpolates a min-max pair many times based on a list of z-scores
    
    This function is to be applied to a pandas DataFrame as follows:
    df22.apply(test2,result_type='expand',axis=1,zs=zs)
    
    Input:
        *series* (Panda Series) : a row of a df containing a min and max score
        *zs* (list) : a list of z-scores, for example drawn from a normal distribution
        *sdvs* (optional) : the amount of sdvs the min and max values are away from the mean
    
    Returns:
        *series* (Panda Series) : the same panda series, but with the interpolated values
                                  added as new columns
    """
    #creates interpolater object that also extrapolates
    #x = the known range; y = the known damages
    
    #optionally: build that the function creates the series of zs itself
    interpolator = interp1d(x=[-1*sdvs,sdvs],y=series.values,kind='linear',fill_value='extrapolate')
    values = interpolator(zs)
    series = series.append(pd.Series(data=values))
    
    return series

def pick_tuples_randint(series,low,high):
    """
    From a series of tuples, randomly return one of the tuple element positions for each tuple
    This is the same position for each tuple (but different for each row in the df you take)
    
    Arguments:
        *series* (Panda Series) : a row from the dataframe, containing multiple tuple values
        *low_bound* (integer) : inclusive lower boundary from which to draw the position
        *high* (integer) : exclusive upper boundary from which to draw the position
    """
    pos = np.random.randint(low,high)
    series = series.apply(lambda x: pick_tuple(x,pos))
    return series

def zs_random_100():
    """Return a list of 100 z-scores drawn from the Guassian Normal Distribution
    
    Todo: replace this by a quasi-random series
    
    This is just one random outcome of:
    zs = np.random.normal(0,1,(100))
    """
    return np.array([
       -1.81904904,  1.43346153, -2.01556988,  0.33519238, -1.10144545,
       -1.01124017, -1.0332434 , -0.36405516, -0.28111336,  0.10429122,
        1.40735649,  1.28344103,  1.3972128 , -0.15406543, -1.47174265,
       -0.10147744, -0.34024692, -0.40890453,  1.12756601,  0.20862727,
       -0.3253316 ,  0.25703024,  2.79110541, -0.17033205, -1.41913996,
       -0.65448251,  0.51201731,  0.77231799,  0.17032116, -0.70288893,
       -0.63874365,  0.76129971,  1.61795471,  0.08258017, -0.05343008,
        0.35984622,  0.59993363,  0.01961202,  1.40584314, -0.61665141,
        0.15245386, -1.17050188, -0.11157971, -1.55225322,  0.59947228,
       -0.39416129, -0.53479878, -0.81968132, -1.79740196,  0.58573876,
        1.11512429,  0.44405585, -0.01129808,  0.25622288,  0.19229016,
       -2.46536852, -0.50192305, -0.67077208,  1.38558846,  0.61440455,
        0.57763725, -0.85932267, -0.73600602,  0.22579569,  0.48226559,
        0.7202608 ,  1.04858219, -0.19894003, -0.70796578,  0.12666693,
       -1.97377998, -1.94367892, -1.11376296, -3.01503489,  0.89284043,
       -1.5776689 , -0.43461092,  1.76601896,  0.14932378,  2.44052397,
        1.34862093, -0.00322519,  0.42097969,  0.77846056, -0.36243739,
       -0.48619513,  0.87939814, -0.18551697, -0.76531307,  0.35014853,
        1.31999189,  0.54934268,  0.08261485,  1.27076425,  1.92736417,
       -1.05738975, -1.50405538, -0.65595933, -0.4835964 , -0.44547794])

def zs_random_1000():
    """Return a list of 100 z-scores drawn from the Guassian Normal Distribution
    
    Todo: replace this by a quasi-random series
    
    This is just one random outcome of:
    zs = np.random.normal(0,1,(100))
    """
    return np.array([
       -1.83911159e+00, -1.97946712e+00, -9.75452307e-01,  1.29197326e+00,
        5.36852568e-01,  1.56519178e-01, -1.08171444e+00,  2.35521714e-02,
       -2.18807331e+00,  1.97691424e-01,  1.71460632e+00,  3.52326440e-01,
        1.46341371e+00, -1.33620266e+00,  9.43271637e-01, -5.56790259e-01,
        1.89784270e+00, -6.80552038e-01,  4.48018657e-01, -1.02006683e+00,
        1.05352790e+00,  9.92035365e-01, -1.13744173e+00, -6.25507930e-01,
        1.25396182e+00, -6.80583958e-02, -7.70753312e-01, -6.01906311e-01,
        4.78393941e-01, -1.42843449e+00, -3.25516241e+00, -1.98313085e-01,
        1.28255748e+00,  3.48314973e-01,  5.58325938e-01, -5.09799207e-01,
       -1.29864880e+00, -1.92873364e-01, -2.56883108e+00, -2.78843519e-01,
        1.10425507e+00, -1.55492196e+00,  3.23236885e-01,  1.37788665e+00,
        2.26307434e+00,  1.05815524e-01, -4.53865541e-01,  4.75324334e-01,
        2.01770047e+00, -2.42015269e-01,  7.04358250e-01,  2.33924115e-01,
       -8.92328631e-02,  1.70158747e+00, -8.90974941e-01,  3.13063200e-01,
       -6.38806130e-01, -1.31835039e+00,  8.70142978e-01,  1.09090377e+00,
       -1.21151036e+00,  1.33316379e+00, -7.90946151e-01, -1.26167357e+00,
       -3.74708539e-01,  1.39937130e-01,  2.45971030e-01, -1.16226710e+00,
       -8.04098806e-01,  1.11922032e+00,  1.75436138e+00,  3.88939916e-01,
        1.97886432e+00, -2.18994436e+00, -2.01116613e+00,  6.35912828e-01,
        1.10167565e+00,  2.15858855e-03,  3.38147927e-01,  1.22896514e+00,
       -4.93752058e-01,  5.85347563e-01,  1.97478426e+00,  1.08242721e-01,
       -1.42156725e-01, -1.47164486e+00, -6.90696893e-02, -1.26590450e+00,
        8.31447331e-01, -5.88207246e-01,  9.19081650e-01,  5.04841337e-01,
        2.35349188e+00,  1.98482534e-01,  8.13354878e-01, -7.56580359e-01,
        4.77093701e-01,  5.83196091e-01,  1.43765185e+00, -2.65628467e-01,
       -3.54898243e-01, -7.03544883e-01, -1.72079554e+00, -1.80506474e-01,
       -6.87200747e-01, -2.27325771e-02,  1.31471974e+00, -6.87789289e-01,
       -1.68450070e-01, -6.71517289e-01, -1.65233472e-02,  5.75163659e-01,
        3.11915531e-02,  7.92581878e-01,  6.12817910e-01, -4.24972052e-02,
        2.12278272e+00, -8.79331065e-01, -1.17505683e+00, -4.10589032e-01,
        2.40021492e+00, -7.05851026e-01,  8.20903618e-01,  5.51628091e-01,
       -1.81413667e+00, -1.01826529e+00,  2.00182251e-01,  1.56421335e+00,
       -1.05490705e+00,  1.86393666e-02, -8.48240714e-01,  2.51849566e-01,
       -1.69786117e-01, -2.35621660e-01, -1.05554556e+00,  6.04805371e-01,
        1.20173256e+00, -1.81233514e-01, -5.30664757e-01,  6.28597524e-02,
        7.58689830e-02,  2.97911281e+00, -1.49007738e+00,  1.24718029e+00,
       -9.06694442e-01,  1.48445566e-01, -2.17505177e+00,  3.27059032e-01,
       -2.42545496e+00,  1.90755733e-01, -1.24269406e+00,  2.79906747e-01,
       -1.08926116e+00,  8.31342878e-01,  1.35730884e+00,  5.64093803e-01,
       -7.17734279e-01,  1.46507778e+00,  1.43448225e-01,  1.23663296e+00,
       -5.03893006e-02, -3.70127755e-01,  4.61839951e-01, -1.19600434e+00,
       -4.60313591e-01, -3.47179054e-01,  1.84926941e-01,  2.14452454e+00,
       -6.23097287e-01, -1.77466861e+00,  4.81628139e-01, -4.92082041e-01,
        4.08431165e-01, -1.07639517e+00, -1.09468366e+00, -1.44870974e+00,
       -1.11609604e+00, -2.04591343e-01, -1.63632345e+00,  9.67949816e-01,
       -5.35752223e-01,  2.85036153e-01,  1.64711312e+00, -8.79924789e-01,
       -1.99522962e-02,  4.98204913e-01,  4.92522315e-01,  1.64490305e+00,
       -1.01834472e+00, -4.44293116e-01, -1.06467173e-01, -1.62300095e+00,
       -1.29446566e+00, -6.14211218e-01, -1.88385133e-01, -8.59947148e-01,
        7.03951282e-01, -4.77152457e-01, -8.11658392e-01, -9.63302967e-01,
       -1.06049116e+00,  7.56979382e-01,  1.14251999e+00,  1.07793274e-01,
        7.46622106e-01,  9.03565326e-02, -1.04644943e-01, -9.21017851e-01,
        1.20593917e+00, -4.76495207e-01, -5.19117232e-01, -1.05230433e-02,
       -7.53917554e-02, -1.64532760e+00,  8.77739836e-01,  1.57546461e-01,
       -1.60376034e+00,  4.50828493e-01,  2.30563936e-01,  4.66997132e-01,
       -8.62933236e-01,  6.17911730e-01,  2.04040135e-01,  1.51189330e+00,
        4.09382452e-02,  9.90711956e-01,  1.65647166e+00, -3.09733654e+00,
       -6.68361501e-01, -1.39927590e-01, -3.65337904e-01,  4.98659624e-01,
       -3.33779937e-01,  1.53351217e+00,  1.09173156e+00,  3.27264920e-01,
       -2.39999485e+00, -1.37029653e-01,  4.96810926e-01, -9.52086730e-01,
       -2.05837890e+00,  5.60537728e-01, -1.52021905e+00,  7.08313798e-01,
       -2.82590824e-02,  1.16903828e-01, -1.44769832e+00,  1.80481060e+00,
       -8.67245882e-01, -1.62822817e+00, -1.16605588e-01,  9.13783791e-01,
       -1.82793031e-01,  9.06431833e-01, -6.38178361e-02,  8.21063353e-01,
       -7.01542397e-01,  1.30211322e+00,  6.27248600e-01, -2.40775570e+00,
        9.33036630e-01,  1.22541587e+00,  2.61736998e-02,  3.76083011e-01,
       -2.48092168e+00,  1.67303673e+00,  2.98339037e-01,  8.86249813e-01,
       -4.96180353e-01,  2.93599502e-01,  4.96883147e-02, -9.75761391e-01,
        2.62479116e-01,  6.15634637e-01,  9.89941136e-01,  7.61081323e-01,
        2.25394319e+00,  2.41531463e-01, -4.85614139e-01,  1.22711820e+00,
        3.68454788e-01, -2.25777389e+00, -2.86737166e-01, -8.60097392e-01,
       -1.67896981e+00, -3.83993592e-01, -6.30060148e-01,  1.67573268e+00,
        7.99793318e-01,  8.56280938e-01,  5.67706796e-01, -5.92149483e-01,
       -1.52398693e+00, -5.17255700e-01,  3.69450290e-01,  1.94517456e-01,
       -1.43308474e+00, -1.05927417e+00,  2.35264655e-01, -4.17632483e-01,
        4.27808846e-01,  2.01304718e+00,  1.83901201e+00,  1.19805489e+00,
       -1.02518410e+00, -1.34585006e-01, -7.59841033e-01,  1.28039567e+00,
        1.16157489e+00, -6.52453068e-01, -6.84800660e-01,  8.65903820e-01,
        2.51601868e+00,  1.35065821e-01, -1.47331278e-01, -1.19597900e-01,
        9.48865079e-01, -1.57243051e+00, -7.64895187e-01, -5.29870712e-01,
       -1.63160070e+00, -1.25464502e+00,  2.89696692e-01, -5.05009049e-01,
       -1.04494587e+00, -4.39455807e-02, -7.19579027e-01,  5.53860702e-01,
       -3.48878832e-01, -1.00104147e+00, -4.41549507e-02, -4.84882524e-01,
        1.15543041e+00, -1.52022149e+00,  2.11782397e+00,  6.71588742e-01,
        1.10240702e+00, -1.66314165e+00, -1.89736365e-01, -6.95388545e-01,
        1.93125846e+00,  4.62947961e-01, -1.32590328e+00, -2.14042841e-02,
        6.21147771e-01, -1.48066060e+00,  1.38708419e+00,  1.95150388e+00,
       -7.21352774e-02, -2.28601997e-01, -7.68413616e-01,  7.60902973e-01,
       -2.42953617e+00,  1.44254366e-01, -6.78189953e-02, -2.01925414e+00,
        8.83989737e-01,  1.33147717e+00, -1.05028266e-01,  4.68328722e-01,
       -1.80257731e+00,  2.02274953e-02, -1.37796673e+00,  1.75714472e+00,
       -1.53911038e-01, -9.46599650e-01,  1.68083637e-01,  1.71377089e+00,
       -2.26221195e+00, -1.65161089e+00, -6.37273836e-01,  1.54431700e+00,
       -8.37256783e-01, -9.17024845e-01, -2.39840295e+00, -1.07579905e-01,
        1.50630164e-01,  2.85112144e-01,  2.31369017e+00, -4.57992953e-01,
        5.39061411e-01, -4.71164797e-02, -8.76574251e-01,  6.57257012e-01,
       -5.64810633e-01,  3.89203360e-02,  3.77771910e-01,  1.25587474e-01,
       -1.66364540e+00,  4.97414624e-01, -1.24349625e+00, -1.98098491e-01,
        2.35177741e-01, -1.51885965e+00, -5.05480428e-01,  9.57345188e-01,
       -2.50315806e-01,  1.68521172e+00,  2.01857029e+00,  7.61575790e-02,
       -3.92043039e-01,  5.85308285e-01, -3.48864433e-01, -1.42696299e+00,
       -9.66902889e-01, -7.26263783e-02,  1.32999475e-01, -1.24333159e-01,
        1.54177562e+00, -4.85708280e-01, -1.09928401e-02, -1.26975929e+00,
        1.11976862e+00, -1.63350957e+00, -2.75033640e+00, -1.55741479e+00,
       -4.84714552e-01, -1.92885208e-01,  1.47339119e+00,  1.01077991e+00,
        1.14209411e+00, -2.09444328e-01, -2.58793572e-01,  1.39736496e+00,
        1.07770876e+00,  1.68531461e-02,  1.08115438e-01,  8.73667410e-01,
       -9.35950661e-01, -1.01802760e+00, -1.04902653e-01, -2.30606255e-01,
       -1.26706946e+00, -1.31961815e+00,  1.24550687e-01,  5.20276219e-01,
        7.28409537e-02, -7.86242089e-01, -3.60436997e-01,  1.18227349e+00,
        1.82154352e-01,  3.93714656e-01,  1.32360702e+00, -6.52343262e-02,
       -4.29922028e-01, -1.02597900e+00, -1.24238907e+00, -5.44934096e-02,
       -1.09713616e+00, -5.15326550e-01,  1.94483410e+00,  2.19060671e-02,
       -1.75723244e+00,  8.94487423e-01,  6.33817006e-01,  5.66580869e-01,
       -1.09602011e+00, -1.59906640e-01,  5.83472785e-02, -1.82347286e+00,
       -1.67652649e-01,  8.01753384e-01,  2.23065247e-01, -1.27063363e-01,
        2.23194255e+00, -1.21880566e+00, -1.24330603e-01, -4.45790011e-01,
       -8.93744697e-01, -1.39395033e+00,  3.61182297e-01,  1.79189543e-01,
       -2.16550973e+00,  9.02787425e-01, -5.89424195e-01, -4.10545145e-01,
       -4.09423447e-01, -2.66732622e-01, -2.83553063e-01, -2.45630120e+00,
        5.48217090e-01,  1.19526288e+00,  1.20595283e-01, -7.53707769e-01,
       -3.39177237e-01,  4.71667335e-01,  6.14832893e-01, -5.85864930e-04,
       -2.99805360e-02, -8.68662466e-01, -5.56505449e-01,  9.32998617e-02,
        8.06299419e-02,  1.48338705e+00, -1.18789286e+00,  7.46551191e-01,
       -7.65634364e-01,  1.76148680e+00, -2.36732414e-01,  1.24617932e-01,
        6.62215211e-01,  2.06995460e+00,  7.90035691e-02, -1.40380053e+00,
       -1.88944209e+00, -6.97581010e-01, -5.21653743e-01,  1.26798188e+00,
       -1.21462101e+00,  3.21290365e-01,  4.05739891e-01,  1.92068295e-01,
       -3.48245533e-01, -9.32702195e-02, -1.69267889e+00, -3.64351993e-01,
       -2.04541367e+00,  2.31312943e-01,  1.87028551e-01, -6.20760653e-01,
       -8.33549954e-02, -9.95979128e-01,  1.63351579e+00,  5.78571159e-02,
       -1.68789891e+00,  4.90072379e-01,  5.66358216e-01,  7.02500020e-01,
       -1.58926575e+00,  1.14419037e+00, -1.31809580e+00, -1.08579573e+00,
        1.37927154e-01, -2.60596403e-01,  1.73708215e+00,  6.85783433e-01,
        1.43514633e+00, -1.20893147e-01,  1.68011107e+00,  2.19288789e-01,
       -4.67753930e-01,  1.63889349e-01, -8.70491734e-01, -9.55439131e-01,
       -5.05424400e-01,  6.34421689e-01, -6.32979222e-01, -1.49291847e-01,
        9.11613274e-01,  1.53280282e+00, -1.06452905e-01, -5.93645190e-01,
       -4.43892280e-01, -9.66381541e-01,  8.68823197e-01, -5.49488844e-01,
        3.00636190e-02, -8.04307070e-01,  8.75116988e-01,  2.08020453e-01,
        1.20219403e+00,  1.80830469e-01,  7.67302668e-01,  9.30120968e-01,
        1.66216914e+00,  8.15734599e-01, -5.62462298e-01, -8.31664368e-01,
        1.59003709e+00, -9.52089644e-01,  4.00693509e-01,  1.74403915e+00,
        1.35916678e-01,  1.08672734e+00,  1.36108365e+00,  1.14191707e+00,
        1.47768710e+00, -5.27009056e-01,  7.11190459e-01,  8.75444127e-01,
        7.46562395e-01, -1.27609124e+00,  1.49835431e-01, -1.64883934e-01,
       -1.02386002e+00, -4.90337180e-01, -4.39294696e-01,  3.02819276e-02,
       -4.04901828e-01,  1.06891962e+00, -9.58324498e-01,  9.08699799e-01,
       -8.28872532e-01, -6.85416789e-02,  6.01736807e-01,  2.77744082e-01,
        1.09268870e+00,  7.39211609e-01, -5.00132510e-01,  1.61271123e+00,
       -9.52208488e-01,  1.46366633e-01,  9.76544447e-01, -5.13977725e-01,
        7.08832663e-01, -1.48691807e+00, -1.35239527e+00, -9.05378925e-01,
       -8.98894964e-01, -1.01020679e-01,  2.53002010e-01, -1.99545427e-01,
       -2.03296240e-01,  9.62716187e-02,  7.88229316e-01,  5.24975718e-01,
        1.96028295e-01,  7.13702048e-03,  6.47258697e-01, -5.84958783e-02,
        4.33201112e-01,  5.16665083e-01,  1.06624767e+00, -1.33623543e-01,
       -3.84555890e-01,  8.18036926e-02,  4.92444522e-01, -1.37498936e-02,
       -9.72380255e-01,  1.48661469e+00,  1.71252677e+00,  1.19521569e+00,
        7.65506567e-01,  2.03768223e-01, -2.96056721e+00, -9.79801568e-01,
       -1.31573220e+00, -9.64891794e-01, -2.22296667e-01, -9.31164966e-01,
        1.37082524e+00,  7.44872745e-01,  7.28753749e-01, -9.02889231e-01,
        3.20284598e-01,  1.08708462e+00, -5.58634040e-01,  3.72527413e-01,
        3.02829968e-01,  1.31631094e+00,  2.62574517e+00,  7.00952821e-01,
       -9.88491198e-01,  2.77690073e-01, -4.29312262e-02, -6.07100349e-01,
       -2.38632851e-01,  1.28405804e+00, -1.00563590e+00, -6.13455987e-01,
        1.60079073e+00,  3.40462364e-01, -3.04728835e-01, -1.01093290e+00,
        4.02393076e-02, -1.72939656e+00, -1.58171644e-01,  1.40087265e+00,
        1.22303325e+00, -3.70402264e-01,  1.98087429e-01, -1.07794546e+00,
       -1.51203613e-01,  8.07555173e-01,  4.82251035e-02, -2.88242828e-01,
        2.89303819e-01, -1.38600173e+00, -5.16507810e-01,  5.68092548e-01,
        6.75496772e-01,  4.57692961e-01,  8.92005045e-03,  2.10169226e-01,
        4.08704322e-01,  2.01462885e+00,  6.35932330e-02, -2.63980519e+00,
        9.00293926e-02,  2.33759846e-01,  6.40021422e-01,  7.62181508e-01,
        7.45710523e-01, -3.86576148e-01, -1.18364445e+00, -9.95202363e-01,
       -1.95951565e-01,  9.44264774e-01, -6.72919603e-01, -3.63802325e-01,
       -1.41191283e+00,  4.90628414e-03,  6.40342870e-01, -5.80492111e-01,
       -1.04793366e+00,  1.93999350e+00, -1.54320457e-01,  1.81234500e-01,
        1.19971372e+00,  5.70028429e-01,  1.99287337e-01,  3.24479644e+00,
        1.53656427e-01,  1.81815370e-01, -3.55002841e-01,  1.25233922e-01,
       -1.62460454e+00,  3.60004537e-01, -2.11674436e-01,  7.46099202e-01,
        4.97822180e-02,  1.35576135e+00, -3.07791744e-01,  6.20278121e-01,
        1.54770994e+00,  8.01115599e-02, -2.17643276e-01, -9.28381802e-01,
       -6.59682426e-01, -6.18128489e-01,  5.89276414e-01,  2.60590607e+00,
        9.40176404e-01, -1.62372152e+00,  2.13977215e-01, -3.43008647e-01,
        1.45075368e-01,  2.09507085e-01,  2.26954908e-01, -1.22942583e+00,
        9.86779127e-01, -2.67182715e-01, -3.83823737e-01,  5.95097652e-01,
        2.78905610e-01,  1.00337039e+00, -2.87027240e-01,  5.57816749e-01,
        4.07775705e-02, -2.15092669e+00,  1.24060535e-01,  4.74491812e-02,
       -1.13990579e+00, -1.48478608e+00, -1.91912610e+00, -1.74809944e+00,
       -4.24801231e-01, -1.78565515e-02,  2.50788227e-01, -8.29793635e-01,
        2.52207870e+00,  1.25096882e+00,  1.49162727e+00,  2.53155437e-01,
        1.31389081e+00, -1.64239179e-01, -7.49068805e-01, -9.05582385e-01,
       -7.38819498e-01, -8.09859137e-01,  4.90754006e-01,  2.74643460e-01,
       -5.50288645e-02, -2.48171085e+00, -4.82405770e-01,  1.56293394e+00,
        5.28310209e-01, -1.10735268e+00, -1.10080451e+00, -1.52925751e+00,
       -4.58747022e-01, -9.59169288e-01, -1.51141759e+00,  7.45565302e-01,
       -5.58634753e-01, -7.30430946e-01,  1.86993771e-01,  7.15270850e-01,
        7.66367823e-01, -9.57874720e-01, -1.49351232e+00, -2.47696453e-01,
        2.40658557e+00, -4.81296172e-01, -4.54466604e-01, -3.80785451e-01,
        5.75832967e-01,  4.63340086e-01, -1.21895762e+00, -5.56534129e-01,
       -1.13642747e+00, -8.30434699e-01,  5.50771579e-01,  4.40989416e-01,
        3.30111432e-01,  1.75651346e-01, -9.87204326e-01, -1.52067982e+00,
        1.16368358e+00, -1.83875901e-01,  1.00323977e+00,  1.12291993e+00,
        8.71678947e-02, -7.31886234e-02,  3.65343877e-01,  4.06864449e-01,
       -7.50392648e-01, -1.79974131e+00, -7.16791344e-01, -3.06729163e-01,
        1.46248228e+00, -1.42033747e-01, -4.44893830e-02,  1.96711743e+00,
        8.67818236e-01,  4.08084011e-01,  6.45804883e-01, -1.37177314e+00,
        2.26075791e+00, -1.36231903e+00,  3.46284035e-01,  1.69988501e+00,
       -8.51793020e-01,  1.25398380e+00,  2.56928007e-01,  1.56037985e-01,
       -6.77793851e-01, -3.00628685e+00, -1.04746834e+00,  1.11456740e-01,
       -8.44908492e-02, -4.74693281e-01,  1.31841602e-01, -5.15341249e-01,
        8.01986038e-02,  7.41865189e-01,  5.76729040e-03, -1.46128626e+00,
        2.19283490e+00,  2.24525346e+00, -3.02249255e-01, -1.52230961e+00,
       -2.18546373e+00,  8.35528757e-01,  3.37975808e-01, -6.99772370e-01,
        6.02040436e-01, -8.73767878e-02, -1.15021225e+00,  1.84566734e-01,
       -4.95754451e-02, -7.61582863e-02, -6.18704724e-01,  1.31994998e+00,
        2.81060329e-01,  6.69853642e-01,  1.81775907e+00,  1.48021492e+00,
        1.27880310e+00,  2.00273379e-01, -1.31632169e+00, -5.24340691e-01,
        6.87963915e-01, -1.01156187e+00, -4.61694011e-01,  1.28434762e+00,
       -2.43878150e-01, -5.87994469e-01,  8.53805974e-02,  1.10815181e+00,
        7.98005466e-02,  1.15736500e+00, -2.02063208e-01,  7.30033178e-01,
       -9.96236297e-02, -2.04170796e+00, -1.03929747e-01,  1.63800491e+00,
        3.56812744e-01,  4.89067210e-02,  4.93054102e-01,  1.12903971e+00,
       -1.01288312e-01, -1.03194273e+00, -1.55690606e+00, -4.08723025e-01,
       -9.39347913e-02,  6.98498003e-01,  4.39044205e-01, -4.56006209e-01,
        9.88105298e-01,  1.80698976e-01, -8.86818703e-02, -1.02087553e+00,
       -2.28646948e-01,  5.06677851e-01, -7.03223776e-01,  1.57436040e-02,
       -7.79550257e-01, -8.47475679e-01, -3.52743352e-01, -1.87646513e-01,
       -4.33190062e-01,  1.24884466e+00,  1.04459627e+00,  6.04405428e-02,
        1.10244651e+00, -1.45671060e-01, -2.85328835e-01,  2.31382819e-01,
        5.96048594e-01,  1.10102134e+00,  1.51392359e-01,  1.56525982e-01,
        6.10266204e-01,  6.82138371e-01,  2.64735235e-01, -1.16814167e+00,
       -1.74445575e+00, -8.57296452e-01, -5.80068853e-01,  1.74395172e+00,
        4.17081689e-02,  3.55792969e-01,  1.58935427e-01,  5.33264807e-01,
       -4.78170935e-01,  7.48807491e-01, -1.31057448e-01,  1.72987483e-01,
        1.41205710e+00, -4.48597674e-01, -5.05041390e-01,  9.55923564e-01,
       -1.44559301e+00, -2.20238570e-01, -9.73797653e-01, -5.59192009e-01,
        3.09181846e-01, -4.68003041e-02,  1.39071966e+00,  7.76594073e-01,
       -1.96010329e-01, -1.82040876e+00, -3.52293902e-01, -3.75091570e-01,
       -7.40766008e-01, -1.01282982e+00,  1.14671873e-01, -1.21553278e-01,
       -3.99130289e-01,  2.01834122e-01,  4.40330438e-01, -2.26413714e+00,
       -3.34376505e-02, -3.53429820e-01, -1.39122298e+00,  8.11919082e-01,
       -1.61565969e-01, -1.20910997e+00, -3.60606067e-01, -3.70724573e-01,
        4.41100716e-01,  6.40000900e-01,  1.19209529e+00,  7.33850291e-02,
       -1.84737061e+00,  1.80724961e+00, -5.42544836e-01,  5.30402454e-01,
       -9.78460103e-01, -2.60853923e-01,  9.98294085e-01, -1.66211431e+00,
       -2.05373480e+00,  8.11755109e-01,  5.44950519e-01,  8.76729397e-01,
       -1.16197801e+00,  2.60051201e+00,  1.79387164e+00, -1.36730885e+00,
        5.79039465e-01,  1.72913544e-01, -6.15147344e-01,  6.90736698e-01,
       -3.69199212e-01,  1.32590710e+00,  1.55053532e+00,  3.00350448e-02])




def region_background_map(region, **kwargs):
    """
    Fetches and visualizes the main road network for the production of nice background graphics.
    Follows the structure of region_loss_estimation
    
    Arguments:

        *region* (string) -- NUTS3 code of region to consider.
        *log_file* (string) -- name of the logfile.
        
    Updated on 14/10/2019, coordination from fetchclean_total_network.ipynb in folder run_extra
    
    Returns:

        *csv file*
        *pickle*
    
    """
    
    from main_functions import fetch_roads, cleanup_fetch_roads, map_roads, add_default_lanes, load_lane_damage_correction
    from utils_functions import line_length
    
    print("Starting region_background_map for: {}".format(region))
    
    try:
        log_file = kwargs.get('log_file', None) #if no log_file is provided when calling the function, no log will be made
      
        
        # LOAD DATA PATHS - configured in the config.json file
        osm_path = load_config()['paths']['osm_data'] #this is where the osm-extracts are located
        input_path = load_config()['paths']['input_data'] #this is where the other inputs (such as damage curves) are located     
        #hazard_path =  load_config()['paths']['hazard_data'] #this is where the inundation raster are located
        output_path = load_config()['paths']['output'] #this is where the results are to be stored
        
        #For this run; always create a log_file
        if log_file is None:
            log_file = os.path.join(output_path,"region_background_map_{}.txt".format(os.getenv('COMPUTERNAME')))
        
        if log_file is not None: #write to log file
            file = open(log_file, mode="a")
            file.write("\n\nRunning region_loss_estimation for region: {} at time: {}\n".format(region,time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
            file.close()
        
        # SKIP IF REGION IS ALREADY FINISHED
        if os.path.exists(os.path.join(output_path,'total_network','{}.csv'.format(region))): 
            print('{} already finished!'.format(region))
            return None
                
        # IMPORT FLOOD CURVES AND DAMAGE DATA
        map_dam_curves = load_config()['filenames']['map_dam_curves']
        #interpolators = import_flood_curves(filename = map_dam_curves, sheet_name='All_curves', usecols="B:O")
        #dict_max_damages = import_damage(map_dam_curves,"Max_damages",usecols="C:E")
        #max_damages_HZ = load_HZ_max_dam(map_dam_curves,"Huizinga_max_dam","A:G")
        
        # LOAD NUTS REGIONS SHAPEFILE
        NUTS_regions = gpd.read_file(os.path.join(input_path, load_config()['filenames']['NUTS3-shape']))
        
        # EXTRACT ROADS FROM OSM FOR THE REGION
        road_gdf = fetch_roads(osm_path,region,log_file=os.path.join(output_path,'fetch_roads_log_{}.txt'.format(os.getenv('COMPUTERNAME'))))
        
        #Cleanup the road extraction
        road_gdf = cleanup_fetch_roads(road_gdf, region)

        road_gdf['length'] = road_gdf.geometry.apply(line_length)
        road_gdf.geometry = road_gdf.geometry.simplify(tolerance=0.00005) #about 0.001 = 100 m; 0.00001 = 1 m
        road_dict = map_roads(map_dam_curves,'Mapping')
        road_gdf['road_type'] = road_gdf.infra_type.apply(lambda x: road_dict[x]) #add a new column 'road_type' with a more simple classification, based on the detail classification in 'infra_type', using a dict called 'road_dict'
            #all not-known keys should be mapped as 'none' because we use a default dict
        
        # GET GEOMETRY OUTLINE OF REGION
        geometry = NUTS_regions['geometry'].loc[NUTS_regions.NUTS_ID == region].values[0]
               
        # HAZARD STATS
        #hzd_path = os.path.join(hazard_path) 
        #hzd_list = natsorted([os.path.join(hzd_path, x) for x in os.listdir(hzd_path) if x.endswith(".tif")])
        #hzd_names = ['rp10','rp20','rp50','rp100','rp200','rp500']
        
        #hzds_data = create_hzd_df(geometry,hzd_list,hzd_names) #both the geometry and the hzd maps are still in EPSG3035
        #hzds_data = hzds_data.to_crs({'init': 'epsg:4326'}) #convert to WGS84=EPSG4326 of OSM.
        
        # PERFORM INTERSECTION BETWEEN ROAD SEGMENTS AND HAZARD MAPS
        
        #for iter_,hzd_name in enumerate(hzd_names):
         
        # ADD SOME CHARACTERISTICS OF THE REGION
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
        # REMOVED
        
        # SAVE AS CSV AND AS PICKLE
        df.reset_index(inplace=True,drop=True)
        
        # Drop all 'other' and 'track' roads; bit ugly but it works
        df2 = df[df.road_type == 'motorway']
        df2 = df2.append(df[df.road_type == 'trunk'])
        df2 = df2.append(df[df.road_type == 'primary'])
        df2 = df2.append(df[df.road_type == 'secondary'])
        df2 = df2.append(df[df.road_type == 'tertiary'])
        
        df2.to_csv(os.path.join(output_path,'total_network','{}.csv'.format(region)))
        df2.to_pickle(os.path.join(output_path,'total_network','{}.pkl'.format(region)))
        
        print("Region_background_map finished for: {}".format(region))
        
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