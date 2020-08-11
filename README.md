# OSdaMage
This repository contains a model to intersect all of Europe's roads in OpenStreetmap with flood hazard maps, and calculate direct flood damages for each road segment. The results of the model are published in the scientific journal Natural Hazards and Earth System Sciences: https://www.nat-hazards-earth-syst-sci-discuss.net/nhess-2020-104/ .

The computational core of the model is derived from @ElcoK 's GMTRA model (https://github.com/ElcoK/gmtra). The main differences between OSdaMage and GMTRA are:
 - OSdaMage has strongly improved damage functions
 - OSdaMage makes more extensive use of the metadata (road attributes) available in OpenStreetMap
 - The architecture of OSdaMage is developed for book-keeping on the European Unions NUTS-classification, and also relies on EU statistics to improve the damage estimates
 - OSdaMage only focuses on river floods, whereas GMTRA is a multihazard model
 - OSdaMage accounts for uncertainty in flow velocity
 
In the NHESS paper, the OSdaMage model is used to make a comprehensive comparison with grid-based model approaches on the continental (European) scale, using the CORINA and LUISA land cover classifications. This repository contains the code to reproduce the object-based part of the NHESS study, and does not include the grid-based part.

The OSdaMage model was combined with the flood hazard data of the Joint Research Centre; the flood hazard maps used in this work were calculated with the hydrodynamic model LISFLOOD-FP, while the hydrological input was calculated by the hydrological model LISFLOOD, see Alfieri et al. (2014).

# Baseline model
The core model attributes the baseline (no climate change) flood risk data to all road segments and Europe, and carries out a segment-wise damage calculation including an extensive uncertainty analysis.

### Set Anaconda environment
Install the conda environment as specified in the environment.yml file

### Required inputs baseline model
 - A planet file containing a recent 'Europe dump' of OpenStreetMap, downloadable from an OSM mirror, for example: http://download.openstreetmap.fr/ (linked checked on 15 July 2020)
 - River flood data from the EFAS model: *ask Francesco*
 - A raster with flood protection data \*: for example *ask Francesco*
 - Shapefiles of the European NUTS-2016 classification (https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts)
 - Eurostat real GDP per capita per NUTS0 country, and Eurostat GDP per NUTS3-region \*\*\*
 - The OSMconvert executable, downloadable from https://wiki.openstreetmap.org/wiki/Osmconvert, by default OSdaMage expects a file called: 'osmconvert64.exe'
 - Two pickled files containing some summary statistics derived from OSM
      - default_lanes_temp.pkl - with the median number of lanes per country (to complement any missing road segment attributes *Todo: replace by csv?*
      - NUTS3-names.pkl - containing a list of all the NUTS-3 names *TODO: just get this from the shapefile*
 - All information on the damage curves, stored in an Excel sheet named 'Mapping_maxdamage_curves4.xlsx'

Make sure the paths to these files are set correctly in the config.json file.

\* Some preprocessing in GIS may be required to obtain a raster which aligns with the flood risk data.
\*\* Some pc's may have difficulties with some NUTS-3 regions with very complex geometries (notably NO053 and NO071), the shapes of which may be simplified with any GIS software to speed up the calculations
\*\*\* For some NUTS-3 regions this data may be missing, missing values can be interpolated from neighboring regions or preceding years.

### Step 1: preprocessing 

 -> Run *run_core_model/Preproc_split_OSM.ipynb* (calls multiple functions from postproc_functions.py)
 1.1. For each NUTS-3 region, a seperate .poly file is created (and simplified where necessary)
 1.2. For each NUTS-3 region, an OSM extract is made with the help of the poly file, containing all the OSM data in this NUTS-3 region

### Step 2: main model
The main computations are carried out using parallel processing, coordinated in the notebook:
 -> Run *run_core_model/Main_multi.ipynb* (calls multiple functions from *main_functions.py*)
This notebook calls for each NUTS-3 region the function *region_loss_estimation*

*region_loss_estimation* carries out the loss calculation for one NUTS-3 region, as follows:
  1.1. It calls *fetch_roads* which fetches the road network from the .osm.pbf extract of the region
  1.2. It calls *cleanup_fetch_roads* which polishes the road fetch from the .osm.pbf (corrects erratic clipping of NUTS-3 regions completely surrounded by other NUTS-3 regions and cuts roads that extent over the boundary of the NUTS-3 region)\
  1.3. It simplifies road geometries to 0.00005 degree, which is less than 5 m for Europe
  1.4. It simplifies the 'infra_type' attribute by mapping it to 7 main 'road_type' categories (motorways, trunks, primary roads, secondary roads, tertiary roads, other roads, tracks). Mapping settings are defined in *input_data/Mapping_maxdamage_curves.xlsx*
  1.5. It masks and vectorizes the six flood rasters using *create_hz_df*
  1.6. It iterates over all the roads using *intersect_hazard* and adds the the following data to each road segment: total segment length, inundated segment length, average water depth over the inundated part
  1.7. For any segment without lane data, the mode (most frequently occuring number of lanes for that road type in that country) is assigned
  1.8. It iterates over all the roads and all damage curves using *road_loss_estimation* and carries out the damage calculation [a]
 
 [a] *road_loss_estimation* calculates for each road segment, the damage for each combination of inundation raster (6 return periods: RP10, RP20, RP50, RP100, RP200, RP500) and damage curve (7 curves Curve C1-C7): i.e. 42 damages for each road segment. These are added as columns in the GeoPandasDataFrame with road segments. However, rather than calculating one single value for each combination, it also accounts for uncertainty in the max damage estimates. This is done by calculating the minimum, maximum and 3 linearly scaled in-between max damage estimates. As a result, each road segment has 42 damage tuples containg the (min, 25%, 50%, 75%, max) damage estimates.
 
 
### Step 3: postprocessing
 -> Run run_core_model/Post_AoI_FP_multi.ipynb *TODO: include this step in the main model?*
 3.1 This add the AoI (used for climate change analysis) and the Flood Protection data to the model
 
 -> Run run_core_model/Post_Baseline_multi.ipynb
 3.2 This carries out the actual risk calculation


For the climate change module:
 - GDP per capita in different SSPs
 - Climate change data 
 - Area of influence data

### References
van Ginkel, K. C. H., Dottori, F., Alfieri, L., Feyen, L., and Koks, E. E.: Direct flood risk assessment of the European road network: an object-based approach, Nat. Hazards Earth Syst. Sci. Discuss., https://doi.org/10.5194/nhess-2020-104, in review, 2020
Koks, E. E., Rozenberg, J., Zorn, C., Tariverdi, M., Vousdoukas, M., Fraser, S. A., Hall, J. W., & Hallegatte, S. (2019). A global multi-hazard risk analysis of road and railway infrastructure assets. Nature Communications, 10(1), 1–11. https://doi.org/10.1038/s41467-019-10442-3
European Commission, Joint Research Centre (2017):  EFAS rapid flood mapping. European Commission, Joint Research Centre (JRC) [Dataset] PID: http://data.europa.eu/89h/85470f72-9406-4a91-9f1f-2a0220a5fa86

