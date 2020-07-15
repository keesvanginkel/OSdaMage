# OSdaMage
This repository contains a model to intersect all of Europe's roads in OpenStreetmap with flood hazard maps, and calculate direct flood damages for each road segment. The results of the model are published in the scientific journal Natural Hazards and Earth System Sciences: https://www.nat-hazards-earth-syst-sci-discuss.net/nhess-2020-104/ .

The computational core of the model is derived from @ElcoK 's GMTRA model (https://github.com/ElcoK/gmtra). The main differences between OSdaMage and GMTRA are:
 - OSdaMage has strongly improved damage functions
 - OSdaMage makes more extensive use of the metadata (road attributes) available in OpenStreetMap
 - The architecture of OSdaMage is developed for book-keeping on the European Unions NUTS-classification, and also relies on EU statistics to improve the damage estimates
 - OSdaMage only focuses on river floods, whereas GMTRA is a multihazard model
 - OSdaMage accounts for uncertainty in flow velocity
 
In the NHESS paper, the OSdaMage model is used to make a comprehensive comparison with grid-based model approaches on the continental (European) scale, using the CORINA and LUISA land cover classifications. This repository contains the code to reproduce the object-based part of the NHESS study, and does not include the grid-based part.

The OSdaMage is meant to be combined with the flood hazard data of the Joint Research Centre, named 'EFAS rapid flood mapping', created with the 2d hydrodynamic model LISFLOOD-FP, or any comparable model.

# Baseline model
The core model attributes the baseline (no climate change) flood risk data to all road segments and Europe, and carries out a segment-wise damage calculation including an extensive uncertainty analysis.

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
*TODO: CHECK AND LOOK THROUGH THIS CODE*

 -> Run run_core_model/Preproc_split_OSM.ipynb (calls multiple functions from postproc_functions.py)
 1.1 For each NUTS-3 region, a seperate .poly file is created (and simplified where necessary)
 1.2 For each NUTS-3 region, an OSM extract is made with the help of the poly file, containing all the OSM data in this NUTS-3 region

### Step 2: main model
 -> Run run_core_model/Main_multi.ipynb (calls multiple functions from main_functions.py)
 *Todo: describe the step of the main model*
 
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

