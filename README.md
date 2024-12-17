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

This repository contains the complete data processing, analysis, and visualization scripts supporting the research paper: "Multivariate compound events drive historical floods and associated losses along the U.S. East and Gulf coasts", currently under review at [npj Natural Hazards](https://www.nature.com/npjnathazards/).

</br>

**CITATIONS**
>1. Ali, J. (2024). Multivariate compound events drive historical floods and associated losses along the U.S. East and Gulf coasts (Version 1). Zenodo. https://doi.org/10.5281/zenodo.14510116
>2. Javed Ali, Thomas Wahl, Joao Morim et al. Multivariate compound events drive historical floods and associated losses along the U.S. East and Gulf coasts, 26 September 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-5040855/v1]

</br>

## **Description**  
The project analyzes multivariate **compound flood events** by combining hydrometeorological drivers:  
- **Storm Surge**  
- **Precipitation**  
- **River Discharge**  
- **Soil Moisture**  
- **Wave Heights**  

We assess how these drivers contribute to floods and evaluate the resulting **socioeconomic losses** (e.g., property damage, crop loss).  

The scripts perform:  
1. **Data Processing**: Prepares large datasets of climate drivers and flood impacts.  
2. **Flood Impact Analysis**: Quantifies contributions of individual drivers to compound events.  
3. **Visualization**: Generates maps, heatmaps, pie charts, bar plots, and clustering outputs.  
4. **Spatial Analysis**: Identifies regions with similar flood characteristics using hierarchical clustering.  



## **Repository Contents**  

### **1. Data Preprocessing**  
Scripts to clean and preprocess hydrometeorological data and SHELDUS socioeconomic loss datasets:  
- **Storm Surge**: [`1_storm-surge-data-prep.py`](./1_storm-surge-data-prep.py)  
- **Precipitation**: [`2_precip-data-prep.py`](./2_precip-data-prep.py)  
- **River Discharge**: [`3_discharge-data-prep.py`](./3_discharge-data-prep.py)  
- **Soil Moisture**: [`4_soil-moisture-data-prep.py`](./4_soil-moisture-data-prep.py)  
- **Wave Heights**: [`5_wave-data-prep.py`](./5_wave-data-prep.py)  
- **County and Metadata Update**: [`7_update-county_names_fips_codes_states_coasts_df.py`](./7_update-county_names_fips_codes_states_coasts_df.py)  

### **2. Flood Impact Analysis**  
Analyzing county-level flood impacts and contributions of each driver:  
- [`6_counties_flood_impact_analysis.py`](./6_counties_flood_impact_analysis.py)  
- **Driver Contributions**: [`10_create-drivers-contributions-columns-in-final-df.py`](./10_create-drivers-contributions-columns-in-final-df.py)  

### **3. Visualization**  
Scripts to visualize flood drivers and impacts:  
- **Heatmaps**: [`8_heatmaps_for_all_counties.py`](./8_heatmaps_for_all_counties.py)  
- **Pie Charts**: [`9_piecharts_for_compound_events.py`](./9_piecharts_for_compound_events.py)  
- **State-Level Bar Plots**: [`16_barplot-state-level-ce.py`](./16_barplot-state-level-ce.py)  

### **4. Clustering and Spatial Analysis**  
Tools to identify patterns and visualize spatial distributions:  
- **Hierarchical Clustering**:  
   - [`12_hierarchical-clustering.py`](./12_hierarchical-clustering.py)  
   - [`13_visualize-clustering-results.py`](./13_visualize-clustering-results.py)  
- **Spatial Mapping**: [`15_create-maps-for-spatial-distributions.py`](./15_create-maps-for-spatial-distributions.py)  

### **5. Loss Contribution Analysis**  
Analyzing socioeconomic impacts:  
- **Median Loss Analysis**: [`17_median-CF-losses-contributions.py`](./17_median-CF-losses-contributions.py)  
- **Subsetting Final Data**: [`11_subset-final-df-csv.py`](./11_subset-final-df-csv.py)  

---

## **Setup Instructions**  

### **1. Prerequisites**  
Install the following Python libraries:  
```bash
pip install pandas numpy matplotlib seaborn geopandas cartopy scipy tqdm scikit-learn
```

### **2. Directory Structure**  
Organize your working directory as follows:  
```bash
repository/
│-- data/
│   ├── ready-data/                # Raw hydrometeorological and SHELDUS data
│   ├── ready-for-analysis/        # Processed data for analysis
│-- outputs/                       # All visualization and analysis outputs
│-- scripts/                       # Python scripts (current directory)
│-- figures/                       # Output plots and maps
│-- README.md                      # This README file
```

---

## **Usage**  

### **1. Data Preprocessing**  
Run the preprocessing scripts to prepare the raw data:  
```bash
python 1_storm-surge-data-prep.py  
python 2_precip-data-prep.py  
python 3_discharge-data-prep.py  
python 4_soil-moisture-data-prep.py  
python 5_wave-data-prep.py  
```

### **2. Flood Impact Analysis**  
Execute the county-level analysis:  
```bash
python 6_counties_flood_impact_analysis.py  
python 10_create-drivers-contributions-columns-in-final-df.py  
```

### **3. Visualization**  
Generate heatmaps, pie charts, and state-level plots:  
```bash
python 8_heatmaps_for_all_counties.py  
python 9_piecharts_for_compound_events.py  
python 16_barplot-state-level-ce.py  
```

### **4. Clustering and Mapping**  
Perform hierarchical clustering and create spatial maps:  
```bash
python 12_hierarchical-clustering.py  
python 13_visualize-clustering-results.py  
python 15_create-maps-for-spatial-distributions.py  
```

### **5. Loss Analysis**  
Analyze loss contributions and median flood impacts:  
```bash
python 17_median-CF-losses-contributions.py  
python 11_subset-final-df-csv.py  
```

---

## **Outputs**  
- **Heatmaps**: Flood driver contributions per county.  
- **Pie Charts**: Visualizing dominant drivers for compound events.  
- **Bar Plots**: State-level flood event percentages and losses.  
- **Maps**: Spatial distributions of compound flooding metrics.  
- **Clustering Results**: Identification of regions with similar flood characteristics.  

Outputs are saved in the `/figures` and `/outputs` directories.  

---

## **Contributing**  
Contributions are welcome! To suggest improvements or report bugs:  
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/your-feature`).  
3. Commit your changes (`git commit -m "Add feature"`).  
4. Push to the branch (`git push origin feature/your-feature`).  
5. Create a pull request.  

---

## **Citations**  
If you use this repository, please cite the paper:  
```  
Ali, J., Wahl, T., Morim, J., et al.  
"Multivariate compound events drive historical floods and associated losses along the U.S. East and Gulf coasts."  
DOI: 10.21203/rs.3.rs-5040855/v1  
```  

---

## **License**  
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## **Contact**  
For questions, contact:  
**Javed Ali**  
- Email: [javed.ali@ucf.edu](mailto:javed.ali@ucf.edu)  

---

This README is highly detailed, clear, and structured to provide ease of use for collaborators and researchers. Let me know if you'd like additional sections or refinements!
