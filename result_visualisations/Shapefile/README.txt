These shapefiles are supplementary data to van Ginkel, K. C. H., Dottori, F., Alfieri, L., Feyen, L., and Koks, E. E.: Direct flood risk assessment of the European road network: an object-based approach, Nat. Hazards Earth Syst. Sci. Discuss., https://doi.org/10.5194/nhess-2020-104, in review, 2020. 

The Python scripts and input data to reproduce these data can be retrieved from https://github.com/keesvanginkel/OSdaMage

More information:
Kees van Ginkel: kees.vanginkel@deltares.nl
Elco Koks: elco.koks@vu.nl

Road geometries and metadata:
© OpenStreetMap contributors 2019. Distributed under a Creative Commons BY-SA License

Each row contains an OSM road segment, which can be inundated during a flood. The meaning of the columns is as follows:

osm_id : OpenStreetMap identifier of the road object [-]
infra_type : type of infrastructure, equals the OSM highway key [-]
lanes : (estimated) number of lanes [-]
bridge : boolean indicating whether an object is a bridge or not [-]
lit : boolean indicating whether street lighting is present [-]
length : length of road segment in meter [m]
road_type : road type, simplified from infra_type [-]
length_rp10 ... rp500 : length of the inundated part of the segment, per return period [m] ***
val_rp10 ... rp500 : average depth over the inundated part of the segment [cm] ***
NUTS-3 ... NUTS-0 : NUTS-ID of the region in which the segment is located [-]
dam_C1_rp10 ... C6_rp500 : damage to the segment per damage curve (Cx) and return period (rpxxx) [2015-euro] ***
dam_HZ : damage to the segment for Huizinga's (2007) damage curve per return period (rpxxx) [2015-euro] ***
Jongman_FP : return period of river flood protection level [year]
AoI_rp10 ... rp500 : unique identifier to connect to JRC hydrological model [-]
EAD_C1 ... C6 : expected annual damage per damage curve (Cx) [2015-euro/year]
EAD_HZ : expected annual damage for Huizinga's damage function [idem]
EAD_lowflow : expected annual damage under low-flow conditions according to new damage curves (C1-C6) [2015-euro/year]
EAD_highflow : expected annual damage under high-flow conditions according to new damage curves (C1-C6) [2015-euro/year]

When no additional information about flow velocity is available, damage per road segment can be estimated by averaging the last two columns ("EAD_lowflow" + "EAD_higflow")/2




