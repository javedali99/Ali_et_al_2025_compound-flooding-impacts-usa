"""
CODEC Storm Surge and Waterlevel Data Processing

Author: Javed Ali (javed.ali@ucf.edu)
Date: 2023-09-13

Description:
This script processes storm surge data for US coastal areas alongside SHELDUS data.
It reads data, preprocesses it, plots unique locations, calculates and plots percentiles,
performs SHELDUS hazard analysis, and saves all data and plots.

Usage:
Ensure that the required libraries are installed and the source data directories contain the CSV files.
Run the script to perform all tasks and generate output files.
"""

# Import required libraries
import os
import warnings
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set the base path for the data
BASE_PATH = "../data/ready-for-analysis/storm-surge/"


def ensure_directory_exists(directory):
    """
    Ensure that a given directory exists, or create it if it doesn't.

    Args:
    - directory (str): The path of the directory to ensure.
    """
    os.makedirs(directory, exist_ok=True)


def ensure_output_directories_exist():
    """
    Ensure the required output directories exist for storm surge data, or create them if they don't.
    """
    directories = [
        f"{BASE_PATH}plots",
        f"{BASE_PATH}plots/time_series",
        f"{BASE_PATH}plots/unique_locations",
        f"{BASE_PATH}plots/percentiles",
        f"{BASE_PATH}plots/max_surge_with_percentiles",
        f"{BASE_PATH}plots/max_waterlevel_with_percentiles",
        f"{BASE_PATH}unique_locations",
        f"{BASE_PATH}combined_data",
        f"{BASE_PATH}time_series_data",
        f"{BASE_PATH}percentiles_data",
        f"{BASE_PATH}max_values",
        f"{BASE_PATH}percentiles_with_surge_data",
        f"{BASE_PATH}percentiles_with_waterlevel_data",
    ]

    for dir_name in directories:
        ensure_directory_exists(dir_name)


def preprocess_surge_data(filename):
    """
    Preprocess the storm surge data.

    Args:
    - filename (str): Path to the storm surge CSV file.

    Returns:
    - DataFrame: Preprocessed storm surge data.
    """
    df = pd.read_csv(filename, parse_dates=["date"])
    df = df[(df["date"].dt.year >= 1980) & (df["date"].dt.year <= 2018)]

    if df.isna().sum().sum() > 0:
        df.dropna(inplace=True)

    return df


def preprocess_sheldus_data(filename):
    """
    Preprocess the SHELDUS data.

    Args:
    - filename (str): Path to the SHELDUS CSV file.

    Returns:
    - DataFrame: Preprocessed SHELDUS data.
    """
    df = pd.read_csv(filename, parse_dates=["Hazard_start", "Hazard_end"])
    df = df[(df["Hazard_start"].dt.year >= 1980) & (df["Hazard_start"].dt.year <= 2018)]

    return df


def plot_unique_locations_on_map(df):
    """
    Plot unique locations on a map using Stamen terrain and save the plot.

    Args:
    - df (DataFrame): DataFrame containing latitude and longitude columns.
    """
    # stamen_terrain = cimgt.Stamen("terrain-background")
    osm_img = cimgt.OSM()
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(projection=osm_img.crs)
    ax.add_image(osm_img, 9)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    ax.scatter(
        df["longitude"], df["latitude"], color="r", s=15, transform=ccrs.Geodetic()
    )

    ax.set_title(
        f"Locations of Storm Surge Data in US Coastal Areas", fontsize=18, y=1.03
    )
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)

    plt.savefig(
        f"{BASE_PATH}plots/unique_locations/storm_surge_unique_locations_map.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calculate_and_save_timeseries_and_percentiles(df, locations_df):
    """
    Calculate and save individual time series for each unique location in the given DataFrame,
    and then calculate and plot the percentiles for each location.

    Args:
    - df (DataFrame): DataFrame containing storm surge data with columns 'date', 'latitude', 'longitude', 'surge', and 'waterlevel'.
    - locations_df (DataFrame): DataFrame containing unique locations with columns 'latitude', 'longitude', and 'ID'.
    """
    # Print a message indicating the start of the function
    print("Calculating Individual Time Series and Percentiles...")

    for i in tqdm(range(len(locations_df))):
        lat = locations_df["latitude"].iloc[i]
        lon = locations_df["longitude"].iloc[i]
        ids = locations_df["ID"].iloc[i]

        county_loc_df = df[(df["latitude"] == lat) & (df["longitude"] == lon)]

        # Randomization for making all surge and waterlevl values unique in all TS
        np.random.seed(123)
        random_df = lambda x: x + np.random.uniform(0, 1) / (10**7)
        county_loc_df["surge"] = list(map(random_df, county_loc_df["surge"]))

        # Randomization for making all waterlevel values unique in all TS
        county_loc_df["waterlevel"] = list(map(random_df, county_loc_df["waterlevel"]))

        # Save the individual time series for each location
        county_loc_df.to_csv(
            f"{BASE_PATH}time_series_data/storm_surge_TS_{ids}.csv", index=False
        )

        # Plot the time series for surge
        plt.figure(figsize=(15, 10))
        plt.plot(
            county_loc_df["date"],
            county_loc_df["surge"],
            "o-",
            color="blue",
            markerfacecolor="r",
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Storm Surge (mm)", fontsize=12)
        plt.title(
            f"Storm Surge at Latitude {lat} and Longitude {lon}", fontsize=18, y=1.03
        )
        plt.savefig(f"{BASE_PATH}plots/time_series/storm_surge_{ids}.png", dpi=300)
        plt.close()

        # Plot the time series for waterlevel
        plt.figure(figsize=(15, 10))
        plt.plot(
            county_loc_df["date"],
            county_loc_df["waterlevel"],
            "o-",
            color="green",
            markerfacecolor="y",
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Water Level (mm)", fontsize=12)
        plt.title(
            f"Water Level at Latitude {lat} and Longitude {lon}", fontsize=18, y=1.03
        )
        plt.savefig(f"{BASE_PATH}plots/time_series/water_level_{ids}.png", dpi=300)
        plt.close()

        # Sort the values in order for surge
        sorted_surge_df = county_loc_df.sort_values("surge")
        sorted_surge_df.reset_index(inplace=True, drop=True)

        # Sort the values in order for waterlevel
        sorted_waterlevel_df = county_loc_df.sort_values("waterlevel")
        sorted_waterlevel_df.reset_index(inplace=True, drop=True)

        # Calculate quantiles for surge
        quantiles_surge = sorted_surge_df["surge"].quantile(
            q=np.arange(0.05, 1.005, 0.005)
        )
        quantiles_surge_df = pd.DataFrame(quantiles_surge)
        quantiles_surge_df.index.names = ["surge_percentiles"]
        quantiles_surge_df = quantiles_surge_df.reset_index()

        # Calculate quantiles for waterlevel
        quantiles_waterlevel = sorted_waterlevel_df["waterlevel"].quantile(
            q=np.arange(0.05, 1.005, 0.005)
        )
        quantiles_waterlevel_df = pd.DataFrame(quantiles_waterlevel)
        quantiles_waterlevel_df.index.names = ["waterlevel_percentile"]
        quantiles_waterlevel_df = quantiles_waterlevel_df.reset_index()

        # Copy the county_loc_df to a new dataframe for waterlevel and surge
        county_loc_df_waterlevel = county_loc_df.copy()
        county_loc_df_surge = county_loc_df.copy()

        # Combine county_loc_df_surge with their respective quantiles
        county_loc_df_surge["surge_percentiles"] = np.interp(
            county_loc_df_surge["surge"],
            quantiles_surge_df["surge"],
            quantiles_surge_df["surge_percentiles"],
        )

        # Combine county_loc_df_waterlevel with their respective quantiles
        county_loc_df_waterlevel["waterlevel_percentile"] = np.interp(
            county_loc_df_waterlevel["waterlevel"],
            quantiles_waterlevel_df["waterlevel"],
            quantiles_waterlevel_df["waterlevel_percentile"],
        )

        # Save the combined data for surge
        county_loc_df_surge.to_csv(
            f"{BASE_PATH}percentiles_with_surge_data/combined_ts_with_percentiles_{ids}.csv",
            index=False,
        )

        # Save the combined data for waterlevel
        county_loc_df_waterlevel.to_csv(
            f"{BASE_PATH}percentiles_with_waterlevel_data/combined_ts_with_percentiles_{ids}.csv",
            index=False,
        )

        # Save the quantiles for each location for surge
        quantiles_surge_df.to_csv(
            f"{BASE_PATH}percentiles_data/quantiles_surge_{ids}.csv", index=False
        )

        # Save the quantiles for each location for waterlevel
        quantiles_waterlevel_df.to_csv(
            f"{BASE_PATH}percentiles_data/quantiles_waterlevel_{ids}.csv", index=False
        )

        # Plot the quantiles for surge
        plt.figure(figsize=(15, 10))
        plt.plot(
            quantiles_surge_df["surge_percentiles"],
            quantiles_surge_df["surge"],
            "o-",
            color="blue",
            markerfacecolor="r",
        )
        plt.xlabel("Percentiles", fontsize=12)
        plt.ylabel("Storm Surge (mm)", fontsize=12)
        plt.title(
            f"Percentiles of Storm Surge at Latitude {lat} and Longitude {lon}",
            fontsize=18,
            y=1.03,
        )
        plt.savefig(f"{BASE_PATH}plots/percentiles/quantiles_surge_{ids}.png", dpi=300)
        plt.close()

        # Plot the quantiles for waterlevel
        plt.figure(figsize=(15, 10))
        plt.plot(
            quantiles_waterlevel_df["waterlevel_percentile"],
            quantiles_waterlevel_df["waterlevel"],
            "o-",
            color="green",
            markerfacecolor="y",
        )
        plt.xlabel("Percentiles", fontsize=12)
        plt.ylabel("Water Level (mm)", fontsize=12)
        plt.title(
            f"Percentiles of Water Level at Latitude {lat} and Longitude {lon}",
            fontsize=18,
            y=1.03,
        )
        plt.savefig(
            f"{BASE_PATH}plots/percentiles/quantiles_waterlevel_{ids}.png", dpi=300
        )
        plt.close()

    print("Time Series and Percentile Calculation and Saving Completed!")


def max_surge_and_waterlevel_in_hazard_window_and_percentiles(
    sheldus_df, df, locations_df
):
    """
    Find the maximum surge and water level values in each Sheldus hazard time window and their corresponding percentiles
    for each unique location.

    Args:
    - sheldus_df (DataFrame): DataFrame containing Sheldus data.
    - df (DataFrame): DataFrame containing storm surge data with columns 'date', 'longitude', 'latitude', 'surge', and 'waterlevel'.
    - locations_df (DataFrame): DataFrame containing unique locations with columns 'latitude', 'longitude', and 'ID'.
    """

    # Print a message indicating the start of the function
    print("Calculating Maximum Values in Hazard Window and their Percentiles...")

    def max_value_in_window(sheldus_df, df_values, window_size, column):
        """
        Extract the maximum value within the Sheldus hazard window for a specified column, extended by a specified number of days.

        Args:
        - sheldus_df (DataFrame): DataFrame containing Sheldus data.
        - df_values (DataFrame): DataFrame with values for a specific location.
        - window_size (int): Number of days to extend the hazard window.
        - column (str): The column to extract maximum values from.

        Returns:
        - DataFrame: Extracted maximum values within the extended hazard window.
        """
        output_df = pd.DataFrame(columns=["Date", "latitude", "longitude", column])
        for i in range(len(sheldus_df)):
            start_date = sheldus_df["Hazard_start"].iloc[i]
            end_date = sheldus_df["Hazard_end"].iloc[i]
            first_window_date = start_date - timedelta(days=window_size)
            last_window_date = end_date + timedelta(days=window_size)
            window_dates = pd.date_range(first_window_date, last_window_date)
            window_values = df_values.loc[df_values.index.intersection(window_dates)]
            max_value = window_values[
                window_values[column] == window_values[column].max()
            ]
            output_df = pd.concat([output_df, max_value], axis=0)
        return output_df

    # Convert columns to datetime
    sheldus_df["Hazard_start"] = pd.to_datetime(sheldus_df["Hazard_start"])
    sheldus_df["Hazard_end"] = pd.to_datetime(sheldus_df["Hazard_end"])

    df["Date"] = df["date"].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    for i in tqdm(range(len(locations_df))):
        lat = locations_df["latitude"].iloc[i]
        lon = locations_df["longitude"].iloc[i]
        ids = locations_df["ID"].iloc[i]

        county_loc_df = df[(df["latitude"] == lat) & (df["longitude"] == lon)]

        # Randomization for making all surge values unique in all TS
        np.random.seed(123)
        random_df = lambda x: x + np.random.uniform(0, 1) / (10**7)
        county_loc_df["surge"] = list(map(random_df, county_loc_df["surge"]))

        # Randomization for making all waterlevel values unique in all TS
        county_loc_df["waterlevel"] = list(map(random_df, county_loc_df["waterlevel"]))

        for column in ["surge", "waterlevel"]:
            sorted_df = county_loc_df.sort_values(column)
            sorted_df.reset_index(inplace=True)

            quantiles = sorted_df[column].quantile(q=np.arange(0.05, 1.005, 0.005))
            quantiles_df = pd.DataFrame(quantiles)
            quantiles_df.index.names = [f"{column}_Percentiles"]
            quantiles_df = quantiles_df.reset_index()

            max_values_df = max_value_in_window(sheldus_df, county_loc_df, 1, column)
            max_values_quantile_df = max_values_df.copy()
            max_values_quantile_df.reset_index(inplace=True)

            quant_interp = np.interp(
                max_values_quantile_df[column],
                quantiles_df[column],
                quantiles_df[f"{column}_Percentiles"],
            )
            quant_interp = pd.DataFrame(quant_interp)
            quant_interp.columns = [f"{column}_Percentiles"]
            max_values_quantile_df = pd.concat(
                [max_values_quantile_df, quant_interp], axis=1
            )

            plt.figure(figsize=(15, 10))
            plt.plot(
                max_values_quantile_df[f"{column}_Percentiles"],
                max_values_quantile_df[column],
                "o",
                color="blue",
                markerfacecolor="r",
            )
            plt.xlabel(f"{column.title()} Percentiles", fontsize=12)
            plt.ylabel(column.title(), fontsize=12)
            plt.title(
                f"Maximum {column.title()} with its Percentiles at Latitude {lat} and Longitude {lon}",
                fontsize=18,
                y=1.03,
            )
            plt.savefig(
                f"{BASE_PATH}/plots/max_{column}_with_percentiles/max_{column}_with_Percentiles_{lat}_{lon}_{ids}.png",
                dpi=300,
            )
            plt.close()

            # Remove the index column
            max_values_quantile_df.drop(columns=["index"], inplace=True)

            max_values_quantile_df.to_csv(
                f"{BASE_PATH}/max_values/Percentiles_of_max_{column}_{lat}_{lon}_{ids}.csv",
                index=False,
            )

    print("Maximum Values in Hazard Window and Percentiles Calculation Completed!")


def main():
    """
    Main function to orchestrate the processing of US coastal storm surge data alongside SHELDUS data.
    """
    ensure_output_directories_exist()

    # Preprocess storm surge data
    surge_data = preprocess_surge_data(
        "../data/ready-data/surge-us-coastline-all-locations-dailyMax-1979-2018.csv"
    )

    # Preprocess SHELDUS data
    sheldus_data = preprocess_sheldus_data(
        "../data/ready-data/sheldus_gulf_east_coasts_counties_flood_df.csv"
    )

    # Create dataframe of unique locations with unique IDs
    unique_locations = (
        surge_data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    )
    unique_locations["ID"] = unique_locations.index + 1
    unique_locations.to_csv(
        f"{BASE_PATH}unique_locations/storm_surge_unique_locations.csv", index=False
    )

    # Plot unique locations on map
    plot_unique_locations_on_map(unique_locations)

    # Create individual time series for each location and calculate and plot percentiles
    calculate_and_save_timeseries_and_percentiles(surge_data, unique_locations)

    # Find the maximum storm surge and water level values in each Sheldus hazard time window and their corresponding percentiles
    max_surge_and_waterlevel_in_hazard_window_and_percentiles(
        sheldus_data, surge_data, unique_locations
    )


# Run the main function
if __name__ == "__main__":
    main()

# Print a message indicating successful completion of the script
print("Script Completed Successfully!")

# End of script
