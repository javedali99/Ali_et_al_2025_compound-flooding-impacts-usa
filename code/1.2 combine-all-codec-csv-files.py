"""
Combine all CSV files in a specific folder into one DataFrame

Author: Javed Ali
Email: javed.ali@ucf.edu
Date: 09/29/2023

"""

# Import required libraries
import glob

import pandas as pd
from tqdm import tqdm

# Set the base path for the output
BASE_PATH = "../data/ready-for-analysis/storm-surge/"


def combine_all_locations_data(dir_path, output_filename):
    """
    Combine all locations data into one DataFrame.

    Args:
    - dir_path (str): Path to the directory containing individual CODEC waterlevel-percentile CSV files.
    - output_filename (str): Name of the output CSV file to save combined data for the coast.

    Notes:
    The function prints the path where the combined file is saved.
    """
    # Fetch all CSV files within the max_values directory of the storm-surge folder
    csv_files = glob.glob(dir_path + "/*.csv")

    # Initialize a list to store dataframes
    dfs = []

    # Loop through each CSV file, read its content, and append to the dfs list
    for csv in csv_files:
        data = pd.read_csv(csv)
        dfs.append(data)

    # Concatenate all dataframes in dfs list
    df_all = pd.concat(dfs, ignore_index=True)

    # Sort the combined DataFrame by the "Date" column
    if "date" in df_all.columns:
        df_all_sorted = df_all.sort_values(by="date")
    else:
        print("Warning: No 'date' column found in the DataFrame. Data is not sorted.")
        df_all_sorted = df_all

    # Remove any "time" column if present
    if "time" in df_all_sorted.columns:
        df_all_sorted.drop("time", inplace=True, axis=1)

    # Rename the column "Date" to "Date_{}".format(water_level) or "Date_{}".format(surge), and latitude and longitude columns
    # to "latitude_{}".format(water_level) or "latitude_{}".format(surge) and "longitude_{}".format(water_level) or "longitude_{}".format(surge)
    # respectively based on the directories percentiles_with_waterlevel_data and percentiles_with_surge_data folders
    if "percentiles_with_surge_data" in dir_path:
        df_all_sorted.rename(
            columns={
                "date": "Date_surge",
                "latitude": "latitude_surge",
                "longitude": "longitude_surge",
                "waterlevel": "waterlevel_surge",
                "waterlevel (tide)": "waterlevel (tide)_surge",
            },
            inplace=True,
        )
    elif "percentiles_with_waterlevel_data" in dir_path:
        df_all_sorted.rename(
            columns={
                "date": "Date_waterlevel",
                "latitude": "latitude_waterlevel",
                "longitude": "longitude_waterlevel",
                "surge": "surge_waterlevel",
                "waterlevel (tide)": "waterlevel (tide)_waterlevel",
            },
            inplace=True,
        )
    else:
        print(
            "Warning: No 'percentiles_with_surge_data' or 'percentiles_with_waterlevel_data' directory found in the path. Data is not renamed."
        )

    # Save the combined and sorted DataFrame to the designated output file
    df_all_sorted.to_csv(output_filename, index=False)

    # Print a message indicating successful combination and saving of data
    print(f"Data Combined and Saved to {output_filename}!")

    # Return the combined DataFrame
    return df_all_sorted


# Combine the data in max_values directory for CODEC water level data
waterlevel_dir_path = f"{BASE_PATH}/percentiles_with_waterlevel_data/"
waterlevel_output_filename = f"{BASE_PATH}/combined_ts_waterlevel-percentiles.csv"
waterlevel_combined_df = combine_all_locations_data(
    waterlevel_dir_path, waterlevel_output_filename
)

# Combine the data in max_values directory for CODEC surge data
surge_dir_path = f"{BASE_PATH}/percentiles_with_surge_data/"
surge_output_filename = f"{BASE_PATH}/combined_ts_surge-percentiles.csv"
surge_combined_df = combine_all_locations_data(surge_dir_path, surge_output_filename)

# Combine both water level and surge data into one DataFrame with all columns without common columns
combined_df = pd.concat([waterlevel_combined_df, surge_combined_df], axis=1)

# Create a column "Date" by copying "Date_waterlevel" column
combined_df["Date"] = combined_df["Date_waterlevel"]

# Save the combined DataFrame to a CSV file
combined_output_filename = (
    f"{BASE_PATH}/combined_ts_waterlevel-surge-percentiles_final.csv"
)
combined_df.to_csv(combined_output_filename, index=False)

# Print a message indicating successful combination and saving of data
print("Script Completed Successfully!")
