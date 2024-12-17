"""
Flooding Events Analysis and Visualization Script

Author: Javed Ali
E-mail: javed.ali@ucf.edu
Date: November 28, 2023

Overview:
This script is designed to analyze and visualize flooding events and their environmental drivers (like precipitation, river discharge, soil moisture,
storm surge, and wave height) across counties in the US Gulf of Mexico and Atlantic coasts. It focuses on generating insightful donut charts for each
county, showcasing the impacts of these events.

The script performs the following tasks:
- Data Processing: Reads and processes flooding event data from a CSV file, covering comprehensive details on each flooding
event, such as start and end dates, property and crop damage, injuries, fatalities, and key climate variables.
- Compound Event Analysis: Utilizes the 'check_compound_events' function to estimate the frequency and severity of compound flood events, including associated losses.
- Visualization: Creates donut charts using matplotlib, illustrating the contribution of different factors to flooding events.
- Customization and Clarity: Adjusts visual elements for clarity, including color maps and text properties. Handles missing data for comprehensive visual representation.
- Iterative Processing: Processes data county by county, generating individual visualizations for localized analysis.
- Robust Error Handling: Includes error management to ensure smooth execution.

Output:
The script produces detailed donut charts for each county, providing a visual understanding of the contributions of various factors on flooding events.

"""

# Import libraries

import glob

# System operations
import os

# Hide warnings
import warnings

# Dealing with date and time
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data wrangling and manipulation
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
from matplotlib.colors import ListedColormap

# Progress bar
from tqdm import tqdm

warnings.filterwarnings("ignore")

########################################################################################################################
##################################### DATA VISUALIZATION FOR FLOODING EVENTS & ITS DRIVERS #############################
########################################################################################################################


def check_compound_events(flood_df, threshold, hazard, hazard_clim_vars):
    """
    Estimates the number of compound flood events based on various percentile thresholds for different
    hydrometeorological variables. It also computes the associated losses and how often each variable
    contributes to compound events.

    Parameters:
    - flood_df (DataFrame): The dataset containing flood event information.
    - threshold (float): The percentile threshold used to determine if a variable contributes to a compound event.
    - hazard (str): The type of hazard being analyzed.
    - hazard_clim_vars (list): A list of climate variables associated with the hazard to be analyzed.

    Returns:
    - A tuple of counts of each variable contributing to compound events and total losses associated with these events.
    """

    # Select hazard-related entries
    sheldus_hazard = flood_df[flood_df["Hazard"].str.contains(hazard)]
    hazard_vars = sheldus_hazard[
        hazard_clim_vars
    ].copy()  # Use a copy to avoid SettingWithCopyWarning

    # Define a dictionary to hold the names of the variables and their corresponding analysis columns
    analysis_vars = {
        "Percentiles_precip": "var1",
        "Percentiles_discharge": "var2",
        "Percentiles_soil_moisture": "var3",
        "surge_percentiles": "var4",
        "waveHs_percentiles": "var5",
    }

    # Filter only available variables in the dataset
    available_vars = [var for var in analysis_vars.keys() if var in flood_df.columns]

    # Initialize sum_thresh column to count the number of threshold exceedances for each event
    hazard_vars["sum_thresh"] = 0

    # Assign a value of '1' for each variable exceeding the threshold and update sum_thresh
    for var in available_vars:
        col_name = analysis_vars[var]
        hazard_vars[col_name] = hazard_vars[var].apply(
            lambda x: 1 if x > threshold else 0
        )
        hazard_vars["sum_thresh"] += hazard_vars[col_name]

    # Calculate total losses
    total_losses = {
        "property_damage": hazard_vars["PropDmgAdj"].sum(),
        "crop_damage": hazard_vars["CropDmgAdj"].sum(),
        "injuries": hazard_vars["Injuries"].sum(),
        "fatalities": hazard_vars["Fatalities"].sum(),
    }

    # Calculate number and details of compound events
    nb_CEs = (hazard_vars["sum_thresh"] >= 2 * threshold).sum()
    CE_df = hazard_vars[hazard_vars["sum_thresh"] >= 2 * threshold]

    # Calculate the number of compound events (where sum_thresh >= 2)
    compound_events = hazard_vars[hazard_vars["sum_thresh"] >= 2 * threshold]

    # Calculate the number of compound events
    nb_compound_events = len(compound_events)

    # Calculate frequency of each variable in compound events
    frequency_counts = {
        var: CE_df[col_name].sum()
        for var, col_name in analysis_vars.items()
        if col_name in CE_df
    }

    # Calculate losses due to compound events
    compound_losses = {
        "property_damage": CE_df["PropDmgAdj"].sum(),
        "crop_damage": CE_df["CropDmgAdj"].sum(),
        "injuries": CE_df["Injuries"].sum(),
        "fatalities": CE_df["Fatalities"].sum(),
    }

    # Calculate percentages of compound events and losses
    total_events = len(hazard_vars)
    percentage_CE = 100 * nb_CEs / total_events if total_events else 0
    loss_percentages = {
        "property_damage": (
            100 * compound_losses["property_damage"] / total_losses["property_damage"]
            if total_losses["property_damage"]
            else 0
        ),
        "crop_damage": (
            100 * compound_losses["crop_damage"] / total_losses["crop_damage"]
            if total_losses["crop_damage"]
            else 0
        ),
        "injuries": (
            100 * compound_losses["injuries"] / total_losses["injuries"]
            if total_losses["injuries"]
            else 0
        ),
        "fatalities": (
            100 * compound_losses["fatalities"] / total_losses["fatalities"]
            if total_losses["fatalities"]
            else 0
        ),
    }

    # Output the analysis
    output_info = {
        "threshold": threshold,
        "total_events": total_events,
        "number_of_compound_events_sum": nb_compound_events,
        "total_losses": total_losses,
        "number_of_compound_events": nb_CEs,
        "frequency_counts": frequency_counts,
        "compound_losses": compound_losses,
        "percentage_CE": percentage_CE,
        "loss_percentages": loss_percentages,
        "property_damage": compound_losses["property_damage"],
        "crop_damage": compound_losses["crop_damage"],
        "injuries": compound_losses["injuries"],
        "fatalities": compound_losses["fatalities"],
    }

    # Additional calculations for detailed information
    loss_percentage_compound = {
        key: (
            compound_losses[key] / total_losses[key] * 100 if total_losses[key] else 0
        )
        for key in total_losses
    }
    loss_percentage_all = {
        key: 100 for key in total_losses
    }  # Since the loss percentage for all events is always 100%

    # Prepare rows for the dataframe
    rows = [
        {"info": "threshold", "data": threshold},
        {"info": "total_events", "data": total_events},
        {"info": "number_of_compound_events_sum", "data": nb_compound_events},
        # Total losses for each damage type
        *[
            {"info": f"total_loss_{key}", "data": value}
            for key, value in total_losses.items()
        ],
        {"info": "number_of_compound_events", "data": nb_CEs},
        # Frequency counts for each driver in case of compound events
        *[
            {"info": f"frequency_count_{key}", "data": value}
            for key, value in frequency_counts.items()
        ],
        # Losses for each damage type by compound events
        *[
            {"info": f"compound_loss_{key}", "data": value}
            for key, value in compound_losses.items()
        ],
        {"info": "percentage_CE", "data": percentage_CE},
        # Loss percentages for each damage type by compound events and all events
        *[
            {"info": f"loss_percentage_compound_{key}", "data": value}
            for key, value in loss_percentage_compound.items()
        ],
        # *[{"info": f"loss_percentage_all_{key}", "data": value} for key, value in loss_percentage_all.items()]
    ]

    # Create the dataframe
    df_output_info = pd.DataFrame(rows)

    # Return tuple with frequency counts of each variable in compound events and total losses
    return (
        nb_compound_events,
        total_events,
        frequency_counts,
        total_losses,
        df_output_info,
    )


########################################################################################################################
##################################### IMPLEMENTATION OF DATA VISUALIZATION #############################################
########################################################################################################################

try:
    # Load the county information from a CSV file
    county_dataset_path = "../data/ready-for-analysis/sheldus/county_names_fips_codes_states_coasts_complete_names_updated_final_v2.csv"
    county_info_df = pd.read_csv(
        county_dataset_path,
    )

    # Define the labels for the pie chart
    labels_map = {
        "Percentiles_precip": "Precipitation",
        "Percentiles_discharge": "River Discharge",
        "Percentiles_soil_moisture": "Soil Moisture",
        "surge_percentiles": "Storm Surge",
        "waveHs_percentiles": "Wave Height",  # Corrected column name
    }

    # Initialize an empty dataframe for combined data from all counties for all thresholds (0.90, 0.95, 0.975, 0.99)
    combined_df = pd.DataFrame(
        columns=[
            "county_name",
            "FIPS",
            "state",
            "threshold",
            "total_events",
            "total_loss_property_damage",
            "total_loss_crop_damage",
            "total_loss_injuries",
            "total_loss_fatalities",
            "number_of_compound_events",
            "frequency_count_Percentiles_precip",
            "frequency_count_Percentiles_discharge",
            "frequency_count_Percentiles_soil_moisture",
            "frequency_count_surge_percentiles",
            "frequency_count_waveHs_percentiles",
            "compound_loss_property_damage",
            "compound_loss_crop_damage",
            "compound_loss_injuries",
            "compound_loss_fatalities",
            "percentage_CE",
            "loss_percentage_compound_property_damage",
            "loss_percentage_compound_crop_damage",
            "loss_percentage_compound_injuries",
            "loss_percentage_compound_fatalities",
        ]
    )

    # Process each county one by one
    for index, row in tqdm(county_info_df.iterrows(), total=county_info_df.shape[0]):
        county_name = row["County"]
        FIPS = row["FIPS"]
        state = row["State"]

        # Constructing the paths for the input and output files specific to the current county
        base_path = f"../flood-impact-analysis/{county_name}_{FIPS}/"
        FIGURE_SAVE_PATH = f"{base_path}piecharts/"
        DATA_SAVE_PATH = f"{base_path}piecharts_data/"

        # Define the base path for data and figure save paths
        # base_figure_path = f"{base_path}piecharts/"

        # Ensure the directories exist
        os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)
        os.makedirs(DATA_SAVE_PATH, exist_ok=True)

        # Load the flood data for the current county
        flood_df_path = os.path.join(
            base_path, f"flood_sheldus_df_{county_name}_{FIPS}.csv"
        )
        flood_df = pd.read_csv(flood_df_path)

        # Convert the date column to datetime format
        flood_df["Hazard_start"] = pd.to_datetime(flood_df["Hazard_start"])
        flood_df["Hazard_end"] = pd.to_datetime(flood_df["Hazard_end"])

        # Modify this list to match the actual column names in your dataset
        hazard_clim_vars = [
            "Hazard",
            "Hazard_start",
            "Hazard_end",
            "PropDmgAdj",
            "CropDmgAdj",
            "Injuries",
            "Fatalities",
            # Include only those variables that are available in the dataset
        ] + [
            var
            for var in [
                "Percentiles_precip",
                "Percentiles_discharge",
                "Percentiles_soil_moisture",
                "surge_percentiles",
                "waveHs_percentiles",
            ]
            if var in flood_df.columns
        ]

        thresholds = [0.90, 0.95, 0.975, 0.99]
        dict_flood_data = {}

        # Perform check for compound events at different thresholds and store results in a dictionary
        for thresh in thresholds:
            (
                nb_compound_events,
                total_events,
                frequency_counts,
                total_losses,
                df_output_info,
            ) = check_compound_events(flood_df, thresh, "Flood", hazard_clim_vars)

            # Add the frequency counts to the dictionary
            dict_flood_data[f"thresh_{thresh}"] = frequency_counts

            # Save the dataframe to a CSV file
            df_output_info.to_csv(
                f"{DATA_SAVE_PATH}compound_events_analysis_{county_name}_{FIPS}_thresh_{thresh}.csv",
                index=False,
            )

            # Use pivot_table to reshape df_output_info from long to wide format
            df_output_info_wide = df_output_info.pivot_table(
                index=None, columns="info", values="data", aggfunc="first"
            )
            df_output_info_wide.reset_index(drop=True, inplace=True)

            # Add the county name and FIPS code to the dataframe
            df_output_info_wide["county_name"] = county_name
            df_output_info_wide["FIPS"] = FIPS
            df_output_info_wide["state"] = state

            # Add the threshold to the dataframe
            df_output_info_wide["threshold"] = thresh

            # Append the threshold-specific data to the combined dataframe
            combined_df = pd.concat(
                [combined_df, df_output_info_wide], ignore_index=True
            )

        # Save the combined dataframe as a CSV file
        combined_df.to_csv(
            f"../flood-impact-analysis/combined_flooding_events_analysis_all_counties_final.csv",
            index=False,
        )

        # Prepare the data for the pie charts
        pie_chart_dicts = {}
        for thresh in thresholds:
            pie_chart_data = {
                key: [value]
                for key, value in dict_flood_data[f"thresh_{thresh}"].items()
                if key.startswith("Percentiles")
                or key in ["surge_percentiles", "waveHs_percentiles"]
            }
            pie_chart_data["Hazard"] = ["Flooding"]
            pie_chart_dicts[f"thresh_{thresh}"] = pie_chart_data

        # Create dataframe for pie chart and melt it for all thresholds
        df_melt_flood = {}
        for thresh in thresholds:
            df_flood = pd.DataFrame(pie_chart_dicts[f"thresh_{thresh}"])

            df_melt_flood[f"thresh_{thresh}"] = pd.melt(
                df_flood,
                id_vars="Hazard",
                value_vars=list(pie_chart_dicts[f"thresh_{thresh}"].keys()),
            )

        # Flood labels and colors
        flood_labels = list(pie_chart_dicts["thresh_0.95"].keys())
        flood_labels.remove(
            "Hazard"
        )  # We don't need 'Hazard' as a label in the pie chart

        # Define the colormap and select specific colors
        # cmap = ListedColormap(sns.color_palette("tab20c", len(flood_labels)).as_hex())
        # flood_color = cmap.colors

        # Define the colormap and select specific colors
        cmap = plt.get_cmap("tab20c")
        flood_color = cmap(np.array([14, 6, 0, 9, 4]))

        # Plotting all pie charts
        fig, axs = plt.subplots(2, 2, figsize=(20, 18))
        fig.suptitle(
            f"Flooding Events in {county_name}, {state} ({FIPS})",
            fontsize=24,
            weight="bold",
        )

        for i, thresh in enumerate(thresholds):
            ax = axs[i // 2, i % 2]
            # Call the function and get the number of compound events and total events
            (
                nb_CEs,
                total_events,
                frequency_counts,
                total_losses,
                df_output_info,
            ) = check_compound_events(flood_df, thresh, "Flood", hazard_clim_vars)
            values = df_melt_flood[f"thresh_{thresh}"]["value"].values

            # Handle NaN values
            values = np.nan_to_num(values, nan=0.0)

            # Check if all values are zero (which might happen if they were all NaN)
            if np.sum(values) == 0:
                # Optionally, you can skip drawing the pie chart or handle this case differently
                ax.text(
                    0.5,
                    0.5,
                    "No compound events at 99th",
                    ha="center",
                    va="center",
                    fontsize=20,
                )
                ax.axis("off")
                # remove all the labels and ticks from the axes
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    right=False,
                    left=False,
                    labelleft=False,
                )
                continue

            pie_labels = [
                labels_map[label] for label in flood_labels if label in labels_map
            ]

            ax.pie(
                df_melt_flood[f"thresh_{thresh}"]["value"],
                labels=None,  # pie_labels,
                colors=flood_color,
                startangle=45,
                autopct=lambda x: "{:2.0f}".format(x * sum(values) / 100),
                pctdistance=0.85,
                textprops={"fontsize": 22},
                explode=(values == max(values)) * 0.1,
            )

            # Add a circle at the centre of the pie chart
            my_circle = plt.Circle(
                (0, 0), 0.7, color="white"
            )  # Adding circle at the centre of the pie chart
            ax.add_artist(my_circle)

            ax.set_title(f"Threshold = {thresh}", fontsize=22, weight="bold")
            # Update the text dynamically based on the function results
            ax.text(
                -0.5,
                -0.1,
                f"Total events = {total_events}\nCompound = {nb_CEs}",
                fontsize=22,
            )

        # Add a common legend for all pie charts in the middle of the figure
        fig.legend(
            pie_labels,
            loc="center",
            fontsize=20,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=plt.gcf().transFigure,
        )

        # Save the figure to a specified path
        plt.savefig(
            f"{FIGURE_SAVE_PATH}flood_pie_chart_thresholds_{county_name}_{FIPS}_noBG.png",
            dpi=500,
            bbox_inches="tight",
            transparent=True,  # To make the background transparent
        )

        plt.savefig(
            f"{FIGURE_SAVE_PATH}flood_pie_chart_thresholds_{county_name}_{FIPS}.png",
            dpi=500,
            bbox_inches="tight",
        )

        plt.savefig(
            f"{FIGURE_SAVE_PATH}flood_pie_chart_thresholds_{county_name}_{FIPS}.svg",
            dpi=500,
            bbox_inches="tight",
        )

        plt.close()

        print(f"\nScript completed successfully for {county_name}, {state} ({FIPS})!")


except Exception as e:
    print(e)
    raise Exception(
        "Error occurred during data visualization for flooding events and its drivers!"
    )


# End of script
