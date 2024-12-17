"""
Heatmaps for SHELDUS flooding events and its drivers in different counties on the US Gulf of Mexico and Atlantic coasts.

Author: Javed Ali
E-mail: javed.ali@ucf.edu
Date: November 27, 2023

Description:
This script generates heatmaps for SHELDUS flooding events and their drivers (precipitation, river discharge,
soil moisture, storm surge, and wave height) in different counties on the US Gulf of Mexico and Atlantic coasts.
It takes into account the availability of storm surge and wave data and adjusts the heatmaps accordingly.

"""

# Import libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Hide warnings
import warnings
warnings.filterwarnings("ignore")



# Function to plot the heatmaps
def create_heatmap_for_flood_drivers(
    df,
    clim_vars,
    sort_values_by,
    y_label,
    y_ticks,
    label_text,
    df_save_path,
    fig_title,
    fig_save_path,
    figsize,
):
    """
    Function to select all relevant hydrometeorological variables associated with a hazard and create a heatmap for
    understanding their compounding effects on the hazard.

    Parameters:
    ----------
        df: dataframe (SHELDUS with all climate variables)
        hazard: name of the hazard (str)
        clim_vars: list of all climate variables associated with a particular hazard with property/crop damage, hazard name and duration columns (str list)
        sort_values_by: sort all values based on either property damage or crop damage (str)
        label_text: list of x-axis ticks text (str list)
        df_save_path: path to save the data hazard and its all relevant climate variables
        fig_title: title for the figure (str)
        fig_save_path: path to save the heatmap of hazard and its all relevant climate variables (str with figure type extention i.e. png or svg)

    """

    # Filter the dataframe for the specified hazard and select relevant columns
    hazard_clim_vars = df[df["Hazard"].str.contains("Flood")]
    imp_vars = clim_vars
    hazard_imp_vars = hazard_clim_vars[imp_vars]

    # Save the filtered dataframe to CSV
    hazard_imp_vars.to_csv(df_save_path, index=False)

    # Reorder columns to place the damage type first
    ordered_cols = [sort_values_by] + [col for col in clim_vars if col != sort_values_by]
    hazard_imp_vars = hazard_imp_vars[ordered_cols]

    # drop hazard info columns
    hazard_imp_vars.drop(["Hazard", "Hazard_end"], axis=1, inplace=True)

    # set hazards as index and sort values based on property/crop damage or fatalities/injuries
    heatmap_df = hazard_imp_vars.sort_values(sort_values_by, ascending=False).set_index(
        ["Hazard_start"]
    )


    if sort_values_by == "PropDmgAdj":
        heatmap_df = heatmap_df[heatmap_df["PropDmgAdj"] > 0]

    elif sort_values_by == "CropDmgAdj":
        heatmap_df = heatmap_df[heatmap_df["CropDmgAdj"] > 0]

    elif sort_values_by == "Fatalities":
        heatmap_df = heatmap_df[heatmap_df["Fatalities"] > 0]

    elif sort_values_by == "Injuries":
        heatmap_df = heatmap_df[heatmap_df["Injuries"] > 0]

    # heatmap_df.drop(sort_values_by, axis=1, inplace=True)

    # Adjust colorbar length based on the number of rows
    num_rows_heatmap = len(heatmap_df)
    cbar_len = 0.50 if num_rows_heatmap > 30 else 1  # Shorten colorbar if more than 30 rows

    # visualization
    plt.figure(figsize=figsize)  # (20, 18)
    sns.set(font_scale=2)

    xticks_label = label_text

    heatmap = sns.heatmap(
        heatmap_df,
        cmap="rocket_r",
        annot=True,
        fmt=".4f",
        linewidths=2,
        cbar_kws={
            "label": "Percentiles",
            "orientation": "vertical",
            "shrink": cbar_len,
        },
        vmin=0.95,
        vmax=1,
        xticklabels=xticks_label,
    )
    # linewidths=0.5, "shrink": 0.50, annot_kws={'size':20}, yticklabels=y_ticks_label

    for t in heatmap.texts:
        current_text = t.get_text()

        text_transform = (
            lambda x: f"{round(x / 1000000000, 2)}B"
            if x / 1000000000 >= 1
            else f"{round(x / 1000000, 2)}M"
            if x / 1000000 >= 1
            else f"{round(x / 1000, 2)}K"
            if x / 1000 >= 1
            else f"{x}"
        )
        t.set_text(text_transform(float(current_text)))

    # ax.figure.axes[-1].yaxis.label.set_size(12)
    # ax.figure.axes[-1].set_ylabel('Percentiles', size=14)
    heatmap.set_ylabel(y_label, labelpad=10)  # fontsize=20

    if y_ticks == False:
        heatmap.axes.yaxis.set_ticks([])  # hide y-axis ticks
    elif y_ticks == True:
        plt.yticks(rotation=0, fontsize=24)  # fontsize=20

    # ax.axes.yaxis.set_visible(False) # hide whole axis label, ticks etc
    plt.xticks(rotation=0, fontsize=24)  # fontsize=20,
    heatmap.set_xlabel("Hydrometeorological Drivers", labelpad=20)  # fontsize=20

    ticklabels = [
        heatmap_df.index[int(tick)].strftime("%Y-%m-%d")
        for tick in heatmap.get_yticks()
    ]
    heatmap.set_yticklabels(ticklabels)

    plt.title(fig_title, loc="center", y=1.03)  # fontsize=28,

    # plt.tight_layout()

    plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")

    plt.close()


# Function to determine the relevant climate variables based on data availability
def get_climate_variables(df, base_vars):
    """
    Determines the relevant climate variables for heatmap creation based on the availability of storm surge and wave data.

    Conditions:
    1. If both storm surge and wave data are available, include both.
    2. If only storm surge data is available (and not wave data), include storm surge.
    3. If only wave data is available (and not storm surge), include wave data.

    Parameters:
    ----------
    df : DataFrame
        The dataframe containing flood event data for a specific county.
    base_vars : list of str
        The base list of climate variables that are always included in the heatmap.

    Returns:
    -------
    list of str
        The list of climate variables to be used in the heatmap, adjusted based on data availability.
    """
    clim_vars = base_vars.copy()

    # Check for storm surge data
    has_surge = "surge_percentiles" in df.columns

    # Check for wave data
    has_wave = "waveHs_percentiles" in df.columns

    # Condition 1: Both storm surge and wave data are available
    if has_surge and has_wave:
        clim_vars.extend(["surge_percentiles", "waveHs_percentiles"])

    # Condition 2: Only storm surge data is available
    elif has_surge:
        clim_vars.append("surge_Percentiles")

    # Condition 3: Only wave data is available
    elif has_wave:
        clim_vars.append("waveHs_Percentiles")

    return clim_vars


# Function to process each county
def process_county(county_info):
    """
    Processes a single county by creating heatmaps for property damage, crop damage, fatalities, and injuries.

    Parameters:
    ----------
    county_info : DataFrame row
        A row from the dataframe containing county information (County, FIPS code, State).
    """
    county_name, FIPS, state = county_info["County"], county_info["FIPS"], county_info["State"]
    base_path = f"../flood-impact-analysis/{county_name}_{FIPS}/"
    DATA_SAVE_PATH = f"{base_path}data_heatmaps/"
    FIGURE_SAVE_PATH = f"{base_path}heatmaps/"
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)

    # Load and preprocess flood data
    flood_df_path = os.path.join(base_path, f"flood_sheldus_df_{county_name}_{FIPS}.csv")
    flood_df = pd.read_csv(flood_df_path)
    flood_df["Hazard_start"] = pd.to_datetime(flood_df["Hazard_start"])
    flood_df["Hazard_end"] = pd.to_datetime(flood_df["Hazard_end"])

    base_vars = ["Hazard", "Hazard_start", "Hazard_end", "Percentiles_precip", "Percentiles_discharge", "Percentiles_soil_moisture"]

    # Generate heatmaps for each type of damage
    for damage_type in ["PropDmgAdj", "CropDmgAdj", "Fatalities", "Injuries"]:
        vars = get_climate_variables(flood_df, base_vars + [damage_type])

        # # Generate labels for the heatmap
        # labels = [label_dict.get(var, var) for var in vars if var not in ["Hazard", "Hazard_start", "Hazard_end"]]

        # Filter out unwanted columns and generate labels for the heatmap
        damage_label = label_dict[damage_type]
        labels = [damage_label] + [label_dict.get(var, var) for var in vars if
                                   var not in ["Hazard", "Hazard_start", "Hazard_end", damage_type]]

        # Dynamically determine the figure size
        num_rows = len(flood_df)
        # fig_height = max(4, min(40, num_rows*0.82))  # Adjust the divisor to change row-to-height ratio
        fig_height = num_rows * 0.82

        # Number of rows for the figure if the damage type is not property damage based on the crop damage or fatalities/injuries
        num_rows_else = len(
            flood_df[(flood_df["CropDmgAdj"] > 0) | (flood_df["Fatalities"] > 0) | (flood_df["Injuries"] > 0)])

        # fig_height_else = num_rows_else * 1.75

        # Set figure height for other damage types; use a default height if no rows are found
        if num_rows_else > 0:
            fig_height_else = max(4, num_rows_else * 1.75)
        else:
            fig_height_else = 4  # Default minimum height


        create_heatmap_for_flood_drivers(
            df=flood_df,
            clim_vars=vars,
            sort_values_by=damage_type,
            y_label="Flood Events",
            y_ticks=True,
            label_text=labels,
            df_save_path=f"{DATA_SAVE_PATH}flood_{damage_type}_heatmap_{county_name}_{state}_{FIPS}.csv",
            fig_title=f"{damage_type} for SHELDUS Flooding Events and its Drivers in {county_name}, {state} ({FIPS})",
            fig_save_path=f"{FIGURE_SAVE_PATH}flood_{damage_type}_heatmap_{county_name}_{FIPS}.png",
            figsize=(24, fig_height if damage_type == "PropDmgAdj" else fig_height_else),
        )

    print(f"Script completed successfully for {county_name}, {state} ({FIPS})!")


# Label dictionary for climate variables
label_dict = {
    "PropDmgAdj": "Property \nDamage \n(USD)",
    "CropDmgAdj": "Crop \nDamage \n(USD)",
    "Fatalities": "Fatalities",
    "Injuries": "Injuries",
    "Percentiles_precip": "Precipitation",
    "Percentiles_discharge": "River \nDischarge",
    "Percentiles_soil_moisture": "Soil \nMoisture",
    "surge_percentiles": "Storm \nSurge",
    "waveHs_percentiles": "Wave \nHeight"
}

base_labels = ["Precipitation", "River \nDischarge", "Soil \nMoisture"]

# Read county data and initiate processing
county_dataset_path = "../data/ready-for-analysis/sheldus/county_names_fips_codes_states_coasts_complete_names_updated_final_v2.csv"

counties_df = pd.read_csv(county_dataset_path)

for _, county_info in tqdm(counties_df.iterrows()):
    process_county(county_info)


# End of script