<p align="center">
    <img src="https://github.com/javedali99/compound-flooding-impacts-usa-coastline/assets/15319503/b8cb6c25-3f0f-4788-bbcd-3387c6267e48" alt="disaster" width="150" height="150">
  </a>
  <h1 align="center">Multivariate compound events drive historical floods and associated losses along the U.S. East and Gulf coasts</h1>
</p>

<!--
<br>

<p align="center">
  <img src="https://github.com/javedali99/compound-flooding-impacts-usa-coastline/assets/15319503/89d1da7d-db26-42c9-815d-bdf33a352504" alt="flooding"/></a>
</p>
-->

</br>

This repository contains Python scripts for data processing, analysis, impact assessment, clustering, spatial analysis, and visualization supporting the research paper: "Multivariate compound events drive historical floods and associated losses along the U.S. East and Gulf coasts", currently in press at [npj Natural Hazards](https://www.nature.com/npjnathazards/).

The study investigates compound flood events caused by multiple hydrometeorological drivers (e.g., storm surge, precipitation, river discharge, soil moisture, wave heights) across 203 coastal counties in the U.S. East and Gulf coasts from 1980 to 2018. We quantify flood frequencies, identify dominant drivers, analyze regional patterns, and link these events to socio-economic losses (e.g., property and crop damage).

</br>

**CITATIONS**
>1. Ali, J. (2024). Analysis of historical socio-economic losses driven by multivariate compound flooding events (Version 1). Zenodo. https://doi.org/10.5281/zenodo.14510116

>2. Javed Ali, Thomas Wahl, Joao Morim et al. Multivariate compound events drive historical floods and associated losses along the U.S. East and Gulf coasts, 26 September 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-5040855/v1]

</br>



## **Scripts Description**

### **1. Data Preprocessing**  
These scripts process flood drivers datasets to prepare for analysis:  
- **`1_storm-surge-data-prep.py`**: Prepares storm surge data, computes percentiles, and extracts maxima during hazard windows.  
- **`2_precip-data-prep.py`**: Processes precipitation data, filters relevant time periods, and computes peak rainfall events.  
- **`3_discharge-data-prep.py`**: Prepares river discharge data, resamples to daily maxima, and analyzes key hazard periods.  
- **`4_soil-moisture-data-prep.py`**: Analyzes soil moisture data, calculates percentiles, and integrates with hazard data.  
- **`5_wave-data-prep.py`**: Prepares wave height data, calculates percentiles, and identifies maxima during hazard windows.  



### **2. Flood Impact Analysis**  
Scripts to combine and analyze flooding events and their socio-economic impacts:  
- **`6_counties_flood_impact_analysis.py`**: Analyzes flood driver data at the county level and identifies significant compound events.  
- **`10_create-drivers-contributions-columns-in-final-df.py`**: Computes contributions of each hydrometeorological driver to compound flood events.  
- **`11_subset-final-df-csv.py`**: Filters the final dataset based on thresholds for compound flooding events and excludes irrelevant FIPS codes.  



### **3. Visualization**  
Scripts to visualize the results of flood driver analysis:  
- **`8_heatmaps_for_all_counties.py`**: Generates heatmaps to display driver contributions across all counties.  
- **`9_piecharts_for_compound_events.py`**: Creates pie charts showing the contributions of each driver to flooding events for key counties.  
- **`16_barplot-state-level-ce.py`**: Produces bar plots highlighting the percentage of compound events and losses at the state level.  



### **4. Clustering and Spatial Analysis**  
Scripts to identify spatial patterns and dominant flood drivers:  
- **`12_hierarchical-clustering.py`**: Performs hierarchical clustering to group counties with similar compound flooding characteristics.  
- **`13_visualize-clustering-results.py`**: Generates spatial maps to visualize clustering results.  
- **`14_dominant-drivers-in-clusters.py`**: Summarizes the dominant flood drivers for each identified cluster using pie charts.  
- **`15_create-maps-for-spatial-distributions.py`**: Creates choropleth maps to visualize flood driver contributions and loss metrics across counties.  



### **5. Loss Contribution Analysis**  
Scripts for detailed analysis of property and crop losses due to compound flooding:  
- **`17_median-CF-losses-contributions.py`**: Computes median loss contributions for property and crop damage and generates maps to display the results.  






