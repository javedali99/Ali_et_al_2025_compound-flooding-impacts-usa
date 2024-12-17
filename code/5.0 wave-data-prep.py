"""
Wave Data Processing

Author: Javed Ali (javed.ali@ucf.edu)
Date: 2023-09-19

Description:
This script processes wave data for US coastal areas.
It reads data, preprocesses it, calculates and plots percentiles, 
and saves all data and plots. It also finds the maximum wave height values in each SHELDUS 
hazard time window and their corresponding percentiles for each unique location.

Usage:
Ensure that the required libraries are installed and the source data directories contain the CSV files.
Run the script to perform all tasks and generate output files.
"""

import glob

# Import required libraries
import os
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Base directory for wave data
WAVE_DATA_PATH = "../data/ready-data/WIS/"

# Set the base path for the output
BASE_PATH = "../data/ready-for-analysis/waves/"


def ensure_directory_exists(directory):
    """
    Ensure that a given directory exists, or create it if it doesn't.

    Args:
    - directory (str): The path of the directory to ensure.
    """
    os.makedirs(directory, exist_ok=True)


def ensure_output_directories_exist(region):
    """
    Ensure the required output directories exist for wave data, or create them if they don't.

    Args:
    - region (str): The coastal region being processed (e.g., 'Atlantic' or 'GulfOfMexico').
    """
    directories = [
        f"{BASE_PATH}{region}/plots",
        f"{BASE_PATH}{region}/plots/percentiles",
        f"{BASE_PATH}{region}/plots/max_wave_with_percentiles",
        f"{BASE_PATH}{region}/percentiles_data",
        f"{BASE_PATH}{region}/max_values",
        f"{BASE_PATH}{region}/percentiles_with_wave_data",
    ]

    for dir_name in directories:
        ensure_directory_exists(dir_name)


def preprocess_wave_data(filename):
    """
    Preprocess the wave data.

    Args:
    - filename (str): Path to the wave CSV file.

    Returns:
    - DataFrame: Preprocessed wave data.
    """
    df = pd.read_csv(filename, parse_dates=["time"])
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


def calculate_and_plot_percentiles(df, region, stationID):
    """
    Calculate and plot the percentiles for each station's data.

    Args:
    - df (DataFrame): DataFrame containing wave data for a specific station.
    - region (str): The coastal region being processed (e.g., 'Atlantic' or 'GulfOfMexico').
    - stationID (str): The unique station ID.
    """

    # Randomization for making all waveHs values unique
    np.random.seed(123)
    randomize_values = lambda x: x + np.random.uniform(0, 1) / (10**7)
    df["waveHs"] = list(map(randomize_values, df["waveHs"]))

    # Sort the waveHs values
    sorted_df = df.sort_values("waveHs")
    sorted_df.reset_index(inplace=True, drop=True)

    # Create a new column of Date from time column and change it to datetime
    sorted_df["Date"] = sorted_df["time"].copy()
    sorted_df["Date"] = pd.to_datetime(sorted_df["Date"])

    # Set the index to time
    sorted_df.set_index("time", inplace=True)

    # Convert hourly data to daily maximum values
    sorted_df = sorted_df.resample("D").max()

    # Calculate quantiles for waveHs
    quantiles_wave = sorted_df["waveHs"].quantile(q=np.arange(0.05, 1.005, 0.005))
    quantiles_wave_df = pd.DataFrame(quantiles_wave)
    quantiles_wave_df.index.names = ["waveHs_percentiles"]
    quantiles_wave_df = quantiles_wave_df.reset_index()

    # Copy the sorted wave data to new DataFrame
    wave_df = sorted_df.copy()

    # Reset the index
    wave_df.reset_index(inplace=True)

    # Combine the wave data with their percentiles
    quantiles_interp = np.interp(
        wave_df["waveHs"],
        quantiles_wave_df["waveHs"],
        quantiles_wave_df["waveHs_percentiles"],
    )

    quantiles_interp = pd.DataFrame(quantiles_interp)
    quantiles_interp.columns = ["waveHs_percentiles"]
    wave_df = pd.concat([wave_df, quantiles_interp], axis=1)

    # Save the wave data with percentiles
    wave_df.to_csv(
        f"{BASE_PATH}{region}/percentiles_with_wave_data/combined_ts_with_percentiles_{stationID}.csv",
        index=False,
    )

    # Save the quantiles for the station
    quantiles_wave_df.to_csv(
        f"{BASE_PATH}{region}/percentiles_data/percentiles_wave_{stationID}.csv",
        index=False,
    )

    # Plot the quantiles for waveHs
    plt.figure(figsize=(15, 10))
    plt.plot(
        quantiles_wave_df["waveHs_percentiles"],
        quantiles_wave_df["waveHs"],
        "o-",
        color="blue",
        markerfacecolor="r",
    )
    plt.xlabel("Percentiles", fontsize=12)
    plt.ylabel("Wave Height (m)", fontsize=12)
    plt.title(
        f"Percentiles of Wave Height for Station {stationID}", fontsize=18, y=1.03
    )
    plt.savefig(
        f"{BASE_PATH}{region}/plots/percentiles/percentiles_wave_{stationID}.png",
        dpi=300,
    )
    plt.close()


def max_wave_in_hazard_window_and_percentiles(sheldus_df, df, region, stationID):
    """
    Find the maximum wave height values in each SHELDUS hazard time window and their corresponding percentiles
    for each unique station.

    Args:
    - sheldus_df (DataFrame): DataFrame containing SHELDUS data.
    - df (DataFrame): DataFrame containing wave data for a specific station.
    - region (str): The coastal region being processed (e.g., 'Atlantic' or 'GulfOfMexico').
    - stationID (str): The unique station ID.
    """

    def max_value_in_window(sheldus_df, df_values, window_size, column):
        """
        Extract the maximum value within the SHELDUS hazard window for a specified column, extended by a specified number of days.

        Args:
        - sheldus_df (DataFrame): DataFrame containing SHELDUS data.
        - df_values (DataFrame): DataFrame with values for a specific station.
        - window_size (int): Number of days to extend the hazard window.
        - column (str): The column to extract maximum values from.

        Returns:
        - DataFrame: Extracted maximum values within the extended hazard window.
        """
        output_df = pd.DataFrame(columns=["time", "latitude", "longitude", column])
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

    df["Date"] = df["time"].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("time", inplace=True)
    df.index = pd.to_datetime(df.index)

    # Convert hourly data to daily maximum values
    df = df.resample("D").max()

    # Randomization for making all waveHs values unique
    np.random.seed(123)
    randomize_values = lambda x: x + np.random.uniform(0, 1) / (10**7)
    df["waveHs"] = list(map(randomize_values, df["waveHs"]))

    # Sort the waveHs values
    sorted_df = df.sort_values("waveHs")
    sorted_df.reset_index(inplace=True)

    quantiles = sorted_df["waveHs"].quantile(q=np.arange(0.05, 1.005, 0.005))
    quantiles_df = pd.DataFrame(quantiles)
    quantiles_df.index.names = ["waveHs_Percentiles"]
    quantiles_df = quantiles_df.reset_index()

    max_values_df = max_value_in_window(sheldus_df, df, 1, "waveHs")
    max_values_quantile_df = max_values_df.copy()
    max_values_quantile_df.reset_index(inplace=True, drop=True)

    quant_interp = np.interp(
        max_values_quantile_df["waveHs"],
        quantiles_df["waveHs"],
        quantiles_df["waveHs_Percentiles"],
    )
    quant_interp = pd.DataFrame(quant_interp)
    quant_interp.columns = ["waveHs_Percentiles"]
    max_values_quantile_df = pd.concat([max_values_quantile_df, quant_interp], axis=1)

    plt.figure(figsize=(15, 10))
    plt.plot(
        max_values_quantile_df["waveHs_Percentiles"],
        max_values_quantile_df["waveHs"],
        "o",
        color="blue",
        markerfacecolor="r",
    )
    plt.xlabel("Wave Height Percentiles", fontsize=12)
    plt.ylabel("Wave Height (m)", fontsize=12)
    plt.title(
        f"Maximum Wave Height with its Percentiles for Station {stationID}",
        fontsize=18,
        y=1.03,
    )
    plt.savefig(
        f"{BASE_PATH}{region}/plots/max_wave_with_percentiles/max_wave_with_Percentiles_{stationID}.png",
        dpi=300,
    )
    plt.close()

    # Save the max values and their percentiles
    max_values_quantile_df.to_csv(
        f"{BASE_PATH}{region}/max_values/max_values_{stationID}.csv", index=False
    )


def main():
    """
    Main function to orchestrate the processing of wave data.
    """
    # Define the regions to process
    regions = ["Atlantic", "GulfOfMexico"]

    # Preprocess SHELDUS data
    sheldus_data = preprocess_sheldus_data(
        "../data/ready-data/sheldus_gulf_east_coasts_counties_flood_df.csv"
    )

    for region in regions:
        ensure_output_directories_exist(region)

        # Process each wave data file for the current region
        wave_data_files = glob.glob(f"{WAVE_DATA_PATH}{region}_subset_data_csv/*.csv")
        for file_path in tqdm(wave_data_files):
            # Extract the station ID from the file name
            stationID = os.path.basename(file_path).replace(".csv", "")

            # Preprocess wave data
            wave_df = preprocess_wave_data(file_path)

            # Calculate and plot percentiles for the station's wave data
            calculate_and_plot_percentiles(wave_df, region, stationID)

            # Calculate maximum wave values in the SHELDUS hazard time window and their corresponding percentiles
            max_wave_in_hazard_window_and_percentiles(
                sheldus_data, wave_df, region, stationID
            )


if __name__ == "__main__":
    main()

# Print that the script is completed successfully
print("Script completed successfully.")

# End of script
