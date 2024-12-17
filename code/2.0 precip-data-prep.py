"""
County Hydrometeorological Data Processing

Author: Javed Ali (javed.ali@ucf.edu)
Date: 2023-09-08
Description:
This script processes precipitation and SHELDUS data for multiple counties.
It reads data, preprocesses it, plots unique locations, calculates and plots percentiles,
performs Sheldus hazard analysis, and combines all data into one dataframe.

Usage:
Ensure that the required libraries are installed and the source data directories contain the CSV files.
Run the script to perform all tasks and generate output files.
"""

import glob

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
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")


def ensure_directory_exists(directory):
    """
    Ensure that a given directory exists, or create it if it doesn't.

    Args:
    - directory (str): The path of the directory to ensure.
    """
    os.makedirs(directory, exist_ok=True)


def ensure_county_directories_exist(county_name, fips_code):
    """
    Ensure the required output directories exist for a given county, or create them if they don't.

    Args:
    - county_name (str): Name of the county.
    - fips_code (str): FIPS code of the county.
    """
    base_path = f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/"

    directories = [
        "plots",
        "plots/time_series",
        "plots/unique_locations",
        "plots/percentiles",
        "plots/max_precip_with_percentiles",
        "unique_locations",
        "combined_data",
        "time_series_data",
        "percentiles_data",
        "percentiles_with_precip_data",
        "max_values",
    ]

    for dir_name in directories:
        ensure_directory_exists(base_path + dir_name)


def preprocess_precip_data(filename):
    """
    Preprocesses the precipitation data by performing the following steps:
    1. Reads the precipitation data from a CSV file.
    2. Filters the data to include only the years between 1980 and 2018.
    3. Removes any rows with missing values.

    Args:
        filename (str): Path to the precipitation CSV file.

    Returns:
        DataFrame: Preprocessed precipitation data.

    """
    df = pd.read_csv(filename, parse_dates=["Time"])
    df = df[(df["Time"].dt.year >= 1980) & (df["Time"].dt.year <= 2018)]

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


def plot_unique_locations_on_map(df, county_name, fips_code):
    """
    Plot unique locations on a map using Stamen terrain and save the plot.

    Args:
    - df (DataFrame): DataFrame containing latitude and longitude columns.
    - county_name (str): Name of the county for which the data belongs.
    - fips_code (str): FIPS code of the county.
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
        f"Locations of Precipitation Data in {county_name} County", fontsize=18, y=1.03
    )
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)

    plt.savefig(
        f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/plots/unique_locations/{county_name}_unique_locations_map.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calculate_and_save_timeseries_and_percentiles(
    df, locations_df, county_name, fips_code
):
    """
    Calculate and save individual time series for each unique location in the given DataFrame,
    and then calculate and plot the percentiles for each location.

    Args:
    - df (DataFrame): DataFrame containing precipitation data with columns 'Time', 'longitude', 'latitude', and 'PRCP'.
    - locations_df (DataFrame): DataFrame containing unique locations with columns 'latitude', 'longitude', and 'ID'.
    - county_name (str): Name of the county for which the data belongs.
    - fips_code (str): FIPS code of the county.
    """
    df["latitude"] = df["latitude"].round(6)
    df["longitude"] = df["longitude"].round(6)
    locations_df["latitude"] = locations_df["latitude"].round(6)
    locations_df["longitude"] = locations_df["longitude"].round(6)

    for i in range(len(locations_df)):
        lat = locations_df["latitude"].iloc[i]
        lon = locations_df["longitude"].iloc[i]
        ids = locations_df["ID"].iloc[i]

        county_loc_df = df[(df["latitude"] == lat) & (df["longitude"] == lon)]

        # Create a new column of Date from time column and change it to datetime
        county_loc_df["Date"] = county_loc_df["Time"].copy()
        county_loc_df["Date"] = pd.to_datetime(county_loc_df["Date"])

        # Change index to datetime
        county_loc_df.set_index("Time", inplace=True)

        # Convert hourly data to daily maximum values for each location
        county_loc_df = county_loc_df.resample("D").max()

        # Randomization for making all PRCP values unique in all TS
        np.random.seed(123)
        random_df = lambda x: x + np.random.uniform(0, 1) / (10**7)
        county_loc_df["PRCP"] = list(map(random_df, county_loc_df["PRCP"]))

        # Save the individual time series for each location
        county_loc_df.to_csv(
            f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/time_series_data/{county_name}_TS_{ids}.csv",
            index=False,
        )

        # Plot the time series
        # plt.style.use("seaborn")
        plt.figure(figsize=(15, 10))
        plt.plot(
            county_loc_df["Date"],
            county_loc_df["PRCP"],
            "o-",
            color="blue",
            markerfacecolor="r",
        )
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Precipitation (mm)", fontsize=12)
        plt.title(
            f"Precipitation at Latitude {lat} and Longitude {lon}", fontsize=18, y=1.03
        )
        plt.savefig(
            f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/plots/time_series/{county_name}_{ids}.png",
            dpi=300,
        )
        plt.close()

        # Sort the values in order
        sorted_df = county_loc_df.sort_values("PRCP")

        # Reset the index
        sorted_df.reset_index(inplace=True, drop=True)

        # Calculate quantiles
        quantiles = sorted_df["PRCP"].quantile(q=np.arange(0.05, 1.005, 0.005))

        # Transform data into DataFrame format
        quantiles_df = pd.DataFrame(quantiles)

        # Change the name of the index column and reset the index
        quantiles_df.index.names = ["Percentiles"]
        quantiles_df = quantiles_df.reset_index()

        # Copy the county_loc_df to a new DataFrame
        county_loc_df2 = county_loc_df.copy()

        # Reset the index of county_loc_df
        county_loc_df2.reset_index(inplace=True)

        # Combine county_loc_df precipitation data with their corresponding quantiles
        quant_interp = np.interp(
            county_loc_df2["PRCP"],
            quantiles_df["PRCP"],
            quantiles_df["Percentiles"],
        )

        quant_interp = pd.DataFrame(quant_interp)
        quant_interp.columns = ["Percentiles"]
        county_loc_df2 = pd.concat([county_loc_df2, quant_interp], axis=1)

        # Save the combined DataFrame to a CSV file
        county_loc_df2.to_csv(
            f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/percentiles_with_precip_data/combined_ts_with_percentiles_{county_name}_{fips_code}_{ids}.csv",
            index=False,
        )

        # Save the quantiles for each location
        quantiles_df.to_csv(
            f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/percentiles_data/quantiles_{county_name}_{ids}.csv",
            index=False,
        )

        # Plot the quantiles
        # plt.style.use("seaborn")
        plt.figure(figsize=(15, 10))
        plt.plot(
            quantiles_df["Percentiles"],
            quantiles_df["PRCP"],
            "o-",
            color="blue",
            markerfacecolor="r",
        )
        plt.xlabel("Percentiles", fontsize=12)
        plt.ylabel("Precipitation (mm)", fontsize=12)
        plt.title(
            f"Percentiles of Precipitation at Latitude {lat} and Longitude {lon}",
            fontsize=18,
            y=1.03,
        )
        plt.savefig(
            f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/plots/percentiles/quantiles_{county_name}_{ids}.png",
            dpi=300,
        )
        plt.close()

    print("Time Series and Percentile Calculation and Saving Completed!")


def max_precip_in_hazard_window_and_percentiles(
    sheldus_df, df, locations_df, county_name, fips_code
):
    """
    Find the maximum precipitation value in each Sheldus hazard time window and their corresponding percentiles
    for each unique location.

    Args:
    - sheldus_df (DataFrame): DataFrame containing Sheldus data.
    - df (DataFrame): DataFrame containing precipitation data with columns 'Time', 'longitude', 'latitude', and 'PRCP'.
    - locations_df (DataFrame): DataFrame containing unique locations with columns 'latitude', 'longitude', and 'ID'.
    - county_name (str): Name of the county for which the data belongs.
    - fips_code (str): FIPS code of the county.
    """

    def max_precip_value_in_window(sheldus_df, df_values, window_size):
        output_df = pd.DataFrame(columns=["Date", "latitude", "longitude", "PRCP"])
        for i in range(len(sheldus_df)):
            start_date = sheldus_df["Hazard_start"].iloc[i]
            end_date = sheldus_df["Hazard_end"].iloc[i]
            first_window_date = start_date - timedelta(days=window_size)
            last_window_date = end_date + timedelta(days=window_size)
            window_dates = pd.date_range(first_window_date, last_window_date)
            window_values = df_values.loc[df_values.index.intersection(window_dates)]
            max_value = window_values[
                window_values["PRCP"] == window_values["PRCP"].max()
            ]
            output_df = pd.concat([output_df, max_value], axis=0)
        return output_df

    df["latitude"] = df["latitude"].round(6)
    df["longitude"] = df["longitude"].round(6)
    locations_df["latitude"] = locations_df["latitude"].round(6)
    locations_df["longitude"] = locations_df["longitude"].round(6)

    # Change types of columns to datetime
    sheldus_df["Hazard_start"] = pd.to_datetime(sheldus_df["Hazard_start"])
    sheldus_df["Hazard_end"] = pd.to_datetime(sheldus_df["Hazard_end"])

    # Create a new column of Date from Time column and change it to datetime
    df["Date"] = df["Time"].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Change index to datetime
    df.set_index("Time", inplace=True)
    df.index = pd.to_datetime(df.index)

    for i in range(len(locations_df)):
        lat = locations_df["latitude"].iloc[i]
        lon = locations_df["longitude"].iloc[i]
        ids = locations_df["ID"].iloc[i]

        county_loc_df = df[(df["latitude"] == lat) & (df["longitude"] == lon)]

        np.random.seed(123)
        random_df = lambda x: x + np.random.uniform(0, 1) / (10**7)
        county_loc_df["PRCP"] = list(map(random_df, county_loc_df["PRCP"]))

        sorted_df = county_loc_df.sort_values("PRCP")
        sorted_df.reset_index(inplace=True)

        quantiles = sorted_df["PRCP"].quantile(q=np.arange(0.05, 1.005, 0.005))
        quantiles_df = pd.DataFrame(quantiles)
        quantiles_df.index.names = ["Percentiles"]
        quantiles_df = quantiles_df.reset_index()

        max_precip = max_precip_value_in_window(sheldus_df, county_loc_df, 1)
        max_precip_quantile_df = max_precip.copy()
        max_precip_quantile_df.reset_index(inplace=True)

        quant_interp = np.interp(
            max_precip_quantile_df["PRCP"],
            quantiles_df["PRCP"],
            quantiles_df["Percentiles"],
        )
        quant_interp = pd.DataFrame(quant_interp)
        quant_interp.columns = ["Percentiles"]
        max_precip_quantile_df = pd.concat(
            [max_precip_quantile_df, quant_interp], axis=1
        )

        # plt.style.use("seaborn")
        plt.figure(figsize=(15, 10))
        plt.plot(
            max_precip_quantile_df["Percentiles"],
            max_precip_quantile_df["PRCP"],
            "o",
            color="blue",
            markerfacecolor="r",
        )
        plt.xlabel("Percentiles", fontsize=12)
        plt.ylabel("Precipitation (mm)", fontsize=12)
        plt.title(
            f"Maximum Precipitation with its Percentiles at Latitude {lat} and Longitude {lon}",
            fontsize=18,
            y=1.03,
        )
        plt.savefig(
            f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/plots/max_precip_with_percentiles/max_precip_with_Percentiles_{county_name}_{ids}.png",
            dpi=300,
        )
        plt.close()

        # Remove the index column
        max_precip_quantile_df.drop(columns=["index"], inplace=True)

        max_precip_quantile_df.to_csv(
            f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/max_values/Percentiles_of_max_precip_{county_name}_{ids}.csv",
            index=False,
        )

    print(
        "Maximum Precipitation in Hazard Window and Percentiles Calculation Completed!"
    )


def combine_all_locations_data(dir_path, output_filename):
    """
    Combine all locations precipitation data into one DataFrame for a specific county.

    Args:
    - dir_path (str): Path to the directory containing individual location CSV files for the county.
    - output_filename (str): Name of the output CSV file to save combined data for the county.

    Notes:
    The function prints the path where the combined file is saved.
    """

    # Fetch all CSV files within the max_values directory of the current county
    csv_files = glob.glob(dir_path + "/*.csv")

    # Initialize an empty DataFrame to store combined data
    df_all = pd.DataFrame()

    # Loop through each CSV file, read its content, and append to the combined DataFrame
    for csv in csv_files:
        data = pd.read_csv(csv)
        # df_all = df_all.append(data)
        # Use concat instead of append to avoid the warning. Append is deprecated.
        df_all = pd.concat([df_all, data])

    # Sort the combined DataFrame by the Date column
    df_all_sorted = df_all.sort_values(by="Date")

    # Remove any "index" column if present
    if "index" in df_all_sorted.columns:
        df_all_sorted.drop("index", inplace=True, axis=1)

    # Save the combined and sorted DataFrame to the designated output file
    df_all_sorted.to_csv(output_filename, index=False)

    # Print a message indicating successful combination and saving of data
    print(f"Data Combined and Saved to {output_filename}!")


def process_county_data(precip_file, sheldus_file):
    """
    Process the county data for both hydrometeorological and SHELDUS datasets.

    Args:
    - precip_file (str): Path to the precipitation CSV file for the county.
    - sheldus_file (str): Path to the SHELDUS CSV file for the county.
    """
    # Extract county name and FIPS code from the filename for naming outputs
    split_name = os.path.basename(precip_file).split("_")
    county_name = "_".join(
        split_name[:-2]
    )  # This ensures we get county names with two or more words
    fips_code = split_name[-2]

    ensure_county_directories_exist(county_name, fips_code)

    # Preprocess data
    precip_data = preprocess_precip_data(precip_file)
    sheldus_data = preprocess_sheldus_data(sheldus_file)

    # Create dataframe of unique locations with unique IDs
    unique_locations = (
        precip_data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    )
    unique_locations["ID"] = unique_locations.index + 1
    unique_locations.to_csv(
        f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/unique_locations/{county_name}_unique_locations.csv",
        index=False,
    )

    # Plot unique locations on map
    plot_unique_locations_on_map(unique_locations, county_name, fips_code)

    # Create individual time series for each location and calculate and plot percentiles
    calculate_and_save_timeseries_and_percentiles(
        precip_data, unique_locations, county_name, fips_code
    )

    # Sheldus hazard analysis
    max_precip_in_hazard_window_and_percentiles(
        sheldus_data, precip_data, unique_locations, county_name, fips_code
    )

    # Combine the data in max_values directory for the county
    dir_path = (
        f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/max_values"
    )
    output_filename = f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/combined_max_values_{county_name}_{fips_code}.csv"
    combine_all_locations_data(dir_path, output_filename)

    # Combine the data in percentiles_with_precip_data directory for the county
    ts_dir_path = f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/percentiles_with_precip_data"
    output_filename_ts = f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/combined_ts_with_percentiles_{county_name}_{fips_code}.csv"
    combine_all_locations_data(ts_dir_path, output_filename_ts)


def main():
    """
    Main function to orchestrate the processing of county data.
    """
    # List all CSV files from the precipitation directory
    precip_files = glob.glob("../data/ready-data/precipitation/*.csv")

    for precip_file in tqdm(precip_files):
        # Extract FIPS code from the filename to match with SHELDUS data
        fips_code = os.path.basename(precip_file).split("_")[-2]

        # Find matching SHELDUS file using the FIPS code
        sheldus_files = glob.glob(
            f"../data/ready-data/sheldus-counties/*_{fips_code}.csv"
        )

        if sheldus_files:
            process_county_data(precip_file, sheldus_files[0])


if __name__ == "__main__":
    main()
