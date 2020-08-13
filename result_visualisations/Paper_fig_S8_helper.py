#THIS PYTHON SCRIPT HAS SOME FUNCTIONS FROM PAPER FIGS S8 IPYNB TO HELP THE PARALLEL PROCESSING
import sys
sys.path.append("..") #import folder which is one level higher

import geopandas as gpd
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import os as os
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from utils_functions import load_config
from postproc_functions import *

pp_out_path = r"D:\Europe_trade_disruptions\EuropeFloodResults\Model09_beta\postproc"
baseline_results = os.path.join(pp_out_path,'baseline')   

def prepare_fig_S8(country):
    if not os.path.exists('Paper_fig_S8_{}.png'.format(country)):
        N0 = country
        print(N0)
        #for N0 in NUTS0:
        NUTS3_lst = []
        NUTS3_lst.extend(NUTS_up(N0,True)) #find all the correspondign NUTS-3 regions

        regions = [elem for elem in NUTS3_lst if elem not in NUTS_3_remote(Overseas=True,Creta=True,Spain=True)]
        #print(len(regions))
        ### LOAD AND STRUCTURE THE OSM LIGHTING MIX RESULTS WHICH HAVE ALREADY BEEN POSTPROCESSED
        df = pd.DataFrame()
        exceptions2 = []

        #regions = regions[0:25]

        for region in tqdm(regions):
            try:
                df = df.append(aggregate_byRP(baseline_results,region))
            except Exception as e:
                exceptions2.append(str(region)+str(e))

        df
        #for e in exceptions2:
            #print(e)

        df2 = df.copy().groupby(by='road_type').sum()
        df2 = df2.drop(index='track') #drop the tracks

        df3 = df2.sum(axis=0)
        df3

        #CONVERT THE DATA TO FORMAT SUITABLE FOR PLOTTING AS STACKED BAR PLAT
        index = ['500','200','100','50','20','10']
        data = [] #becomes  a list of dicts
        for rp in [500,200,100,50,20,10]:
            dct = {}
            for boo in ['yes','no']:
                dct[boo] = df3['dam_rp{}_avg_{}'.format(rp,boo)]
            data.append(dct)
        df4 = pd.DataFrame(index=index,data=data)
        df4 = df4*10**(-6)
        df4 = df4.rename(columns={'yes' : 'RP >= FPL', 'no' : 'RP < FPL'})

        #FORMAT STACKED BAR PLOT
        fig, ax = plt.subplots(figsize=(8,4))
        df4.plot(kind='bar',stacked=True, ax=ax, color=['grey','white'], edgecolor='black')

        ax.set_xlabel('Return period (years)')
        ax.set_ylabel('Damage per return period (million Euro)')
        country = NUTS_down(NUTS_down(NUTS_down(df['NUTS-3'][0])))
        fig.suptitle('Country: {}'.format(country))
        fig.savefig('Paper_fig_S8_{}.png'.format(country),bbox_inches='tight',dpi=100)
    else:
        print('{} already finished'.format(country))

def aggregate_byRP(baseline_results, region):
    """Aggregate the results by return period, distinguishing between events below and above the flood protection level
    Used to produce Figure S8.
    
    Arguments:
         *region* (string) -- Name of the NUTS-3 region
    """
    #region = 'DE224'
    df = pd.read_pickle(os.path.join(baseline_results,"{}_EAD_segment_litmix.pkl".format(region)))
    
    all_rp = [10,20,50,100,200,500]
    
    cols_to_drop = ([col for col in df.columns if col.split('_')[0] in ["EAD","AoI",'length','val']]) #drop EAD and AoI cols
    damcols = [col for col in df.columns if col.split('_')[0] == 'dam']
    cols_to_drop.extend([col for col in df.columns if 'dam_HZ' in col])
    df2 = df.copy().drop(columns=cols_to_drop)
    
    df3 = df2.apply(litmix_temp,axis=1)

    to_drop = []
    for CX in ['C1','C2','C3','C4','C5','C6']:
        to_drop.extend([col for col in df3.columns if CX in col])
    df3.drop(columns=to_drop, inplace=True)
    #bridges were already dropped!
    
    #In the remainder of this script we only focus on the average values (ignoring the high-flow and low-flow extremes)
    df4 = df3.copy()
    to_drop = [col for col in df4.columns if 'highflow' in col] 
    to_drop.extend([col for col in df4.columns if 'lowflow' in col])
    df4.drop(columns=to_drop,inplace=True)
    for rp in all_rp:
        rp = str(rp)
        for boo in ['yes','no']: #yes indicates that a flood occurs, no that it does not
            df4['dam_rp{}_avg_{}'.format(rp,boo)] = 0 #add new columns to store the results

    df4 = df4.apply(filter_damage,axis=1)
    
    selcols = ['road_type']
    selcols.extend(['dam_rp' + str(rp) + '_avg_' + boo for rp in [10,20,50,100,200,500] for boo in ['yes','no']])
    df5 = df4[selcols].groupby('road_type').sum()
    df5['NUTS-3'] = df4['NUTS-3'][0]
    
    return df5

def litmix_temp(r):
    """Only used for preparing this figure (for full use, see postproc_functions.py -> lighting_blend())
    It does a litmix and averaging of the damage columns (rather then the EAD columns)
    
    Arguments:
        *r* (road segment) -- a row from the df
    """
    damcols = [x for x in list(r.keys()) if 'dam' in x] #select the damcols
    
    #THE APPROPRIATE TUPLE POSITIONS ARE ALREADY SELECTED
    
    #MIX THE MOST APPROPRIATE DAMAGE CURVES
    if r['road_type'] in ['motorway','trunk']:
        if r['lit'] in ['yes','24/7','automatic','disused']: #missing operating times now
            for rp in ['10','20','50','100','200','500']:
                r['dam_rp{}_lowflow'.format(rp)] = r['dam_C1_rp{}'.format(rp)] #select the expensive (accessories) road
                r['dam_rp{}_highflow'.format(rp)] = r['dam_C2_rp{}'.format(rp)]
                r['dam_rp{}_avg'.format(rp)] = r['dam_rp{}_lowflow'.format(rp)] + r['dam_rp{}_highflow'.format(rp)] / 2
        else:
            for rp in ['10','20','50','100','200','500']:
                r['dam_rp{}_lowflow'.format(rp)] = r['dam_C3_rp{}'.format(rp)] #select the cheap (no accessories) road
                r['dam_rp{}_highflow'.format(rp)] = r['dam_C4_rp{}'.format(rp)]
                r['dam_rp{}_avg'.format(rp)] = r['dam_rp{}_lowflow'.format(rp)] + r['dam_rp{}_highflow'.format(rp)] / 2
    else: #for the other road types
        for rp in ['10','20','50','100','200','500']:
            r['dam_rp{}_lowflow'.format(rp)] = r['dam_C5_rp{}'.format(rp)] #select the cheap (no accessories) road
            r['dam_rp{}_highflow'.format(rp)] = r['dam_C6_rp{}'.format(rp)]
            r['dam_rp{}_avg'.format(rp)] = r['dam_rp{}_lowflow'.format(rp)] + r['dam_rp{}_highflow'.format(rp)] / 2
    
    return r

def filter_damage(r):
    """Filters the damage on a row of a dataframe by comparing it to the flood protection level.
    If the return period of the damage is above (or equal) the Jongman FPL, add to column 'yes' (damage will occur!)
    If the RP of the damage is below the Jongman FPL, add to column 'no' (damage will not occur in reality)
    
    Arguments:
        *r* (road segment) -- a row from the df
    """
    
    FPL = r.Jongman_FP
    
    for rp in [10,20,50,100,200,500]:
        if rp >= FPL:
            r['dam_rp{}_avg_yes'.format(rp)] = r['dam_rp{}_avg'.format(rp)]
        else:
            r['dam_rp{}_avg_no'.format(rp)] = r['dam_rp{}_avg'.format(rp)]
    
    return(r)