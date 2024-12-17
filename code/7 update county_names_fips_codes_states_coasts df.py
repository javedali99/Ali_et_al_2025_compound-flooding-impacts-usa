"""
Flood Impact Analysis Dataset Updater

Author: Javed Ali
Date: November 13, 2023
Email: javed.ali@ucf.edu

Description:
    This script is designed to update a dataset used for analyzing flood impacts. 
    The main dataset, which includes county-level information, is augmented with additional 
    columns that indicate whether corresponding detailed flood impact files contain 
    specific percentile information. These new columns include 'surge_Percentiles_df', 
    'waterlevel_Percentiles_df', and 'waveHs_Percentiles_df'. Each of these columns is 
    populated with 'Yes' or 'No', depending on whether the associated county-specific 
    flood data file contains 'surge_Percentiles', 'waterlevel_Percentiles', or 
    'waveHs_Percentiles' columns, respectively.

    The script iterates over each entry in the main dataset, constructs file paths 
    for corresponding detailed flood data files, reads these files if available, and 
    updates the main dataset based on the presence of the specified columns in these files. 
    The updated dataset is then saved back as a CSV file.

Usage:
    The script is executed with a single argument: the file path of the main dataset. 
    Ensure that the file path is correctly specified and the dataset has the required 
    columns ('County' and 'FIPS') for accurate processing.

Requirements:
    - Python 3.x
    - Pandas library
    - tqdm library

"""

# Import required libraries
import os

import pandas as pd
from tqdm import tqdm


def update_dataset_with_file_check(dataset_path):
    """
    Updates the main dataset with information on the presence of specific columns in
    associated flood data files.

    Parameters:
    - dataset_path (str): The file path of the county_names_fips_codes_states_coasts dataset.

    The function adds three new columns ('surge_Percentiles_df', 'waterlevel_Percentiles_df',
    and 'waveHs_Percentiles_df') to the main dataset and populates them with 'Yes' or 'No'
    based on the presence of corresponding columns in flood data files. The file paths for
    flood data are constructed using 'County' and 'FIPS' columns from the main dataset.
    """
    # Read the main dataset
    try:
        main_dataset_path = os.path.join(
            dataset_path, "county_names_fips_codes_states_coasts.csv"
        )

        main_df = pd.read_csv(main_dataset_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read the main dataset: {e}")

    # Initialize new columns with default value 'No'
    for col in [
        "surge_Percentiles_df",
        "waterlevel_Percentiles_df",
        "waveHs_Percentiles_df",
    ]:
        main_df[col] = "No"

    # Iterate over each row in the main dataset
    for index, row in tqdm(main_df.iterrows()):
        # Check if the county name ends with "_Parish" or "_city"
        county_name = row["County"]
        if not county_name.endswith("_Parish") and not county_name.endswith("_city"):
            # Ensure the county name ends with "_County" if it doesn't end with "_Parish" or "_City"
            if not county_name.endswith("_County"):
                county_name += "_County"

        FIPS = row["FIPS"]
        # Pad the FIPS code to be 5 digits long
        # FIPS = str(row["FIPS"]).zfill(5)
        base_path = f"../flood-impact-analysis/{county_name}_{FIPS}/"
        flood_df_path = os.path.join(
            base_path, f"flood_sheldus_df_{county_name}_{FIPS}.csv"
        )

        # Check if the file exists
        if os.path.exists(flood_df_path):
            try:
                # Read the flood data file
                flood_df = pd.read_csv(flood_df_path)

                # Check for the specified columns and update the main dataset
                if "surge_percentiles" in flood_df.columns:
                    if flood_df["surge_percentiles"].isnull().all():
                        main_df.at[index, "surge_Percentiles_df"] = "No"
                    else:
                        main_df.at[index, "surge_Percentiles_df"] = "Yes"
                if "waterlevel_percentiles" in flood_df.columns:
                    if flood_df["waterlevel_percentiles"].isnull().all():
                        main_df.at[index, "waterlevel_Percentiles_df"] = "No"
                    else:
                        main_df.at[index, "waterlevel_Percentiles_df"] = "Yes"
                if "waveHs_percentiles" in flood_df.columns:
                    if flood_df["waveHs_percentiles"].isnull().all():
                        main_df.at[index, "waveHs_Percentiles_df"] = "No"
                    else:
                        main_df.at[index, "waveHs_Percentiles_df"] = "Yes"
            except Exception as e:
                print(f"Failed to read or process file {flood_df_path}: {e}")

    # Save the updated dataset
    try:
        updated_dataset_path = os.path.join(
            dataset_path, "county_names_fips_codes_states_coasts_updated_v2.csv"
        )
        main_df.to_csv(updated_dataset_path, index=False)
    except Exception as e:
        raise Exception(f"Failed to save the updated dataset: {e}")


def update_dataset_with_complete_county_names(dataset_path):
    """
    Updates the county names in the 'county_names_fips_codes_states_coasts_updated_final' dataset by appending
    appropriate suffixes ('_County', '_Parish', or '_City') to each county name. The function maintains other
    columns and data as is and saves the updated dataset with complete county names.

    This function is specifically designed to standardize county names for consistency in data processing and
    analysis. It checks each county name and appends '_County' if the name does not already end with '_Parish'
    or '_City'. The function ensures that county names are uniformly formatted, facilitating accurate and
    consistent data handling.

    Parameters:
    - dataset_path (str): The file path of the directory where the dataset is stored. It should contain the file
                          'county_names_fips_codes_states_coasts_updated.csv'.

    The function reads the existing 'county_names_fips_codes_states_coasts_updated.csv' file, modifies the
    county names as required, and saves the updated dataset as
    'county_names_fips_codes_states_coasts_complete_names_updated_final.csv' in the same directory.

    """

    # Read the updated main dataset
    try:
        updated_dataset_path = os.path.join(
            dataset_path, "county_names_fips_codes_states_coasts_updated_v2.csv"
        )
        main_df = pd.read_csv(updated_dataset_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read the updated dataset: {e}")

    # Initialize a new DataFrame to store updated county names and other information
    complete_names_df = main_df.copy()

    # Iterate over each row in the updated main dataset
    for index, row in tqdm(main_df.iterrows()):
        # Extract the county name
        county_name = row["County"]

        # Append "_County" if the name does not end with "_Parish" or "_city"
        if not any(suffix in county_name for suffix in ["_Parish", "_city"]):
            if not county_name.endswith("_County"):
                county_name += "_County"

        # Update the complete names DataFrame with the modified county name
        complete_names_df.at[index, "County"] = county_name

    # Save the complete names dataset with updated county names
    try:
        complete_names_dataset_path = os.path.join(
            dataset_path,
            "county_names_fips_codes_states_coasts_complete_names_updated_final_v2.csv",
        )
        complete_names_df.to_csv(complete_names_dataset_path, index=False)
    except Exception as e:
        raise Exception(f"Failed to save the complete names dataset: {e}")


# Execute the script
dataset_path = "../data/ready-for-analysis/sheldus/"
update_dataset_with_file_check(dataset_path)
update_dataset_with_complete_county_names(dataset_path)

# Print a success message
print("The dataset has been successfully updated.")


# End of script
