"""
Author: Javed Ali
Email: javed.ali@ucf.edu
Date: January 20, 2023

Description:
This script is used to perform a flood impact analysis across multiple counties, considering a variety of
variables such as precipitation, soil moisture, river discharge, storm surge, and wave data. The analysis
takes into account the average of percentiles for each flooding event recorded in SHELDUS datbase. The results of the 
analysis are saved in a structured directory format for further inspection.

"""

# Import necessary libraries

import glob

# System operations
import os

# Hide warnings
import warnings

# Dealing with date and time
from datetime import datetime, timedelta

# Mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader
import cartopy.mpl.ticker as cticker

# For calculating distance
import haversine as hs

# Data Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data wrangling and manipulation
import numpy as np
import pandas as pd
import seaborn as sns
from cartopy.io.img_tiles import OSM
from haversine import Unit, haversine, haversine_vector

# Optimizaing the code
from scipy.spatial import cKDTree

# Progress bar
from tqdm import tqdm

warnings.filterwarnings("ignore")

###################################################################################

# Start time of the script
start_time = datetime.now()

# List of states on the Atlantic coast
atlantic_states_abbr = [
    "CT",
    "DE",
    "FL",
    "GA",
    "ME",
    "MD",
    "MA",
    "NH",
    "NJ",
    "NY",
    "NC",
    "RI",
    "SC",
    "VA",
    "PA",
    "DC",
]

# List of states on the Gulf coast
gulf_states_abbr = [
    "AL",
    "FL",
    "LA",
    "MS",
    "TX",
]

# Gulf of Mexico coast counties in Florida
fl_gulf_coast_counties = {
    "Bay_County": "12005",
    "Charlotte_County": "12015",
    "Citrus_County": "12017",
    "Collier_County": "12021",
    "Dixie_County": "12029",
    "Escambia_County": "12033",
    "Franklin_County": "12037",
    "Gulf_County": "12045",
    "Hernando_County": "12053",
    "Hillsborough_County": "12057",
    "Jefferson_County": "12065",
    "Lee_County": "12071",
    "Levy_County": "12075",
    "Liberty_County": "12077",
    "Manatee_County": "12081",
    "Monroe_County": "12087",
    "Okaloosa_County": "12091",
    "Pasco_County": "12101",
    "Pinellas_County": "12103",
    "Santa_Rosa_County": "12113",
    "Sarasota_County": "12115",
    "Taylor_County": "12123",
    "Wakulla_County": "12129",
    "Walton_County": "12131",
}

# Atlantic coast counties in Florida
fl_atlantic_coast_counties = {
    "Brevard_County": "12009",
    "Broward_County": "12011",
    "Clay_County": "12019",
    "Duval_County": "12031",
    "Flagler_County": "12035",
    "Indian_River_County": "12061",
    "Martin_County": "12085",
    "Miami-Dade_County": "12086",
    "Nassau_County": "12089",
    "Palm_Beach_County": "12099",
    "Putnam_County": "12107",
    "St__Johns_County": "12109",
    "St__Lucie_County": "12111",
    "Volusia_County": "12127",
}


###################################################################################
############################ FUNCTIONS ############################################
###################################################################################


def ensure_correct_suffix(county_name):
    """
    Ensure the county name ends with '_County' or '_city' or '_Parish'. If it doesn't end with these, add '_County'.

    Parameters:
    - county_name (str): Name of the county.

    Returns:
    - str: County name with the correct suffix.
    """
    if (
        county_name.endswith("_County")
        or county_name.endswith("_city")
        or "_Parish" in county_name
    ):
        return county_name

    return f"{county_name}_County"


# Extract the county name, FIPS code, and state abbr from the SHELDUS counties data files names
def extract_county_names_fips_codes_states(
    directory_path, SHELDUS_SAVE_PATH, precipitation_data_path
):
    """
    Extracts and aligns county names, FIPS codes, and state abbreviations from filenames in the SHELDUS directory
    and the precipitation data directory. If a FIPS code matches in both directories, the function ensures that the
    county names are updated to match those from the precipitation data files. It also adds a column to store the
    original county names from the SHELDUS data files.

    Parameters:
    - directory_path (str): Path to the directory containing the SHELDUS county data files.
    - SHELDUS_SAVE_PATH (str): Destination path where the resulting CSV will be saved.
    - precipitation_data_path (str): Path to the directory containing the precipitation data files.

    Returns:
    - pd.DataFrame: DataFrame containing columns for 'County', 'FIPS', 'State', and 'Sheldus County Name'.

    Raises:
    - FileNotFoundError: If any of the provided directory paths do not exist or are inaccessible.

    Note:
    This function handles file names with spaces or underscores and ensures the correct alignment of county names,
    state abbreviations, and FIPS codes.
    """

    # Verify the existence of the provided directory paths
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"The SHELDUS directory path {directory_path} does not exist."
        )
    if not os.path.exists(precipitation_data_path):
        raise FileNotFoundError(
            f"The precipitation data directory path {precipitation_data_path} does not exist."
        )

    # Get the list of all files in both directories
    sheldus_files = os.listdir(directory_path)
    precipitation_files = os.listdir(precipitation_data_path)

    # Extract the county names and FIPS codes from precipitation data files
    precipitation_county_info = {
        os.path.splitext(file)[0].split("_")[-1]: " ".join(
            os.path.splitext(file)[0].split("_")[:-1]
        )
        for file in precipitation_files
    }

    # Initialize lists to store the extracted data
    county_names = []
    sheldus_county_names = []  # List to store original SHELDUS county names
    fips_codes = []
    state_abbrs = []

    # Process each file in the SHELDUS directory
    for file in sheldus_files:
        # Normalize the file name to handle spaces and underscores interchangeably
        file_name_parts = file.replace(".csv", "").split("_")

        # The last part of the file name is the FIPS code
        # fips_code = file_name_parts[-1]
        # The last part of the file name is the FIPS code, standardize it to 5 digits
        fips_code = file_name_parts[-1].zfill(
            5
        )  # Pad FIPS code with leading zeros to make it 5 digits
        # The second to last part is the state abbreviation
        state_abbr = file_name_parts[-2]
        # The remaining parts are the county name
        county_name_parts = file_name_parts[:-2]
        county_name = " ".join(county_name_parts)

        # Store the original SHELDUS county name
        sheldus_county_name = county_name

        # Check if the FIPS code matches with any in the precipitation data
        if fips_code in precipitation_county_info:
            # Update the county name with the full name from the precipitation data
            county_name = precipitation_county_info[fips_code]

            # Ensure the updated name ends with '_County' or '_city'
            # county_name = ensure_county_suffix(county_name)

        # Append the extracted information to the lists
        county_names.append(county_name.replace(" ", "_"))
        sheldus_county_names.append(
            sheldus_county_name.replace(" ", "_")
        )  # Format the SHELDUS county name similarly
        fips_codes.append(fips_code)
        state_abbrs.append(state_abbr)

    # Compile the lists into a DataFrame
    county_names_fips_codes_states_df = pd.DataFrame(
        {
            "County": county_names,
            "FIPS": fips_codes,
            "State": state_abbrs,
            "Sheldus County Name": sheldus_county_names,  # Add the new column for SHELDUS county names
        }
    )

    # Ensure all county names have the correct suffix after compiling into a DataFrame
    county_names_fips_codes_states_df["County"] = county_names_fips_codes_states_df[
        "County"
    ].apply(ensure_correct_suffix)

    # Construct the full path for the output CSV file
    output_csv_path = os.path.join(
        SHELDUS_SAVE_PATH, "county_names_fips_codes_states.csv"
    )

    # Save the DataFrame to the CSV file
    county_names_fips_codes_states_df.to_csv(output_csv_path, index=False)

    # Return the DataFrame
    return county_names_fips_codes_states_df


# Create a DataFrame with the county names, FIPS codes, state abbreviations and the coast they belong to.
def create_county_names_fips_codes_states_coasts_df(
    directory_path, SHELDUS_SAVE_PATH, precipitation_data_path
):
    """
    Create a DataFrame with the county names, FIPS codes, state abbreviations and the coast they belong to.

    Parameters:
    directory_path: The path to the directory containing the SHELDUS counties data files
    SHELDUS_SAVE_PATH: The path to the directory where the resulting CSV will be saved
    precipitation_data_path: The path to the directory containing the precipitation data files

    Return:
    A DataFrame with the county names, FIPS codes, state abbreviations and the coast they belong to

    """
    # Get the county names, FIPS codes, and state abbr from the SHELDUS counties data files names
    county_names_fips_codes_states_df = extract_county_names_fips_codes_states(
        directory_path, SHELDUS_SAVE_PATH, precipitation_data_path
    )

    # Create an empty list to store the coast names
    coast_names = []

    # Iterate over each row in the DataFrame
    for index, row in county_names_fips_codes_states_df.iterrows():
        # Get the state abbr
        state_abbr = row["State"]

        # Check if the state abbr is in the list of states on the Atlantic coast
        if state_abbr in atlantic_states_abbr:
            # Add "Atlantic" to the list
            coast_names.append("Atlantic")
        # Check if the state abbr is in the list of states on the Gulf coast
        elif state_abbr in gulf_states_abbr:
            # Add "GulfOfMexico" to the list
            coast_names.append("GulfOfMexico")
        else:
            # Add "Other" to the list
            coast_names.append("Other")

    # Add the coast names to the DataFrame
    county_names_fips_codes_states_df["Coast"] = coast_names

    # Update the coast names for Florida counties based on county name and fips codes and their respective correct coasts
    for county_name, fips_code in fl_atlantic_coast_counties.items():
        county_names_fips_codes_states_df.loc[
            (county_names_fips_codes_states_df["County"] == county_name)
            & (county_names_fips_codes_states_df["FIPS"] == fips_code),
            "Coast",
        ] = "Atlantic"

    for county_name, fips_code in fl_gulf_coast_counties.items():
        county_names_fips_codes_states_df.loc[
            (county_names_fips_codes_states_df["County"] == county_name)
            & (county_names_fips_codes_states_df["FIPS"] == fips_code),
            "Coast",
        ] = "GulfOfMexico"

    # Save the DataFrame as a CSV file
    county_names_fips_codes_states_df.to_csv(
        f"{SHELDUS_SAVE_PATH}county_names_fips_codes_states_coasts.csv", index=False
    )

    return county_names_fips_codes_states_df


# Select the wave data for the county and FIPS code based on the coast they belong to
def select_wave_data(
    county_name, FIPS, county_names_fips_codes_states_coasts_df, wave_data_path
):
    """
    Select the wave data for the county and FIPS code based on the coast they belong to.

    Parameters:
    county_name: The name of the county (with "_County" suffix)
    FIPS: The FIPS code of the county
    county_names_fips_codes_states_coasts_df: A DataFrame with the county names, FIPS codes, state abbreviations and the coast
    wave_data_path: The path to the directory containing the wave data files

    Return:
    A DataFrame with the wave data for the county and FIPS code
    """
    # Get the coast for the county and FIPS code
    coast = county_names_fips_codes_states_coasts_df[
        (county_names_fips_codes_states_coasts_df["County"] == county_name)
        & (county_names_fips_codes_states_coasts_df["FIPS"] == FIPS)
    ]["Coast"].values[0]

    # Read the wave data for the county and FIPS code
    wave_df = pd.read_csv(
        f"{wave_data_path}combined_percentiles_with_wave_data_{coast}.csv",
        parse_dates=["Date"],
    )

    return wave_df


# Ensure that a given directory exists, or create it if it doesn't.
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
    # Constructing the base path
    base_path = f"../flood-impact-analysis/{county_name}_{fips_code}/"

    directories = [
        "plots",
        "data",
    ]

    for dir_name in directories:
        ensure_directory_exists(base_path + dir_name)


# Read SHELDUS data that matches a given county and FIPS code
def read_sheldus_data(county_name, FIPS, sheldus_counties_path):
    """
    Reads SHELDUS data that matches a given county's first word and FIPS code, then creates two datasets:
    one with the full SHELDUS data and another with only "Hazard_start" and "Hazard_end" columns.

    Parameters:
    - county_name (str): The name of the county.
    - FIPS (str): The FIPS code of the county.
    - sheldus_counties_path (str): The path to the directory where SHELDUS data files are stored.

    Returns:
    - sheldus_data: A Pandas DataFrame containing the full SHELDUS data for the matching county and FIPS code.
    - sheldus_dates: A Pandas DataFrame with "Hazard_start", "Hazard_end", and "ID" columns for the matching county and FIPS code.

    Raises:
    - FileNotFoundError: If no files match the county's first word and FIPS code.
    - FileExistsError: If multiple files match the county's first word and FIPS code.

    The function will perform the following steps:
    1. Extract the first word of the county name to match files in the SHELDUS directory.
    2. Construct a file path pattern using the first word of the county and the FIPS code.
    3. Use the glob module to find files that match the pattern, accounting for spaces and underscores.
    4. Handle cases where no file or multiple files are found.
    5. Read the matching file into a DataFrame.
    6. Convert date columns to datetime format and filter data within the desired date range.
    7. Extract a subset of columns for a secondary DataFrame.
    8. Assign a unique ID to each row in both DataFrames.
    9. Return both DataFrames.
    """

    # Extract the first word of the county name for matching files
    first_word_of_county = county_name.split("_")[0]

    # Constructing the file path with wildcard for the first word of the county and FIPS code
    file_pattern = f"{first_word_of_county}*_{FIPS}.csv"
    file_path_pattern = os.path.join(sheldus_counties_path, file_pattern)

    # Using glob to find files that match the pattern, accounting for spaces and underscores interchangeably
    file_pattern_with_spaces = file_pattern.replace("_", " ")
    file_path_pattern_with_spaces = os.path.join(
        sheldus_counties_path, file_pattern_with_spaces
    )

    matching_files = glob.glob(file_path_pattern) + glob.glob(
        file_path_pattern_with_spaces
    )

    # Handle cases where no file or multiple files are found
    if not matching_files:
        raise FileNotFoundError(f"No file matches the pattern: {file_path_pattern}")
    elif len(matching_files) > 1:
        raise FileExistsError(f"Multiple files match the pattern: {file_path_pattern}")

    # Read the matching file into a DataFrame
    sheldus_data = pd.read_csv(matching_files[0])

    # Converting date columns to datetime format
    sheldus_data["Hazard_start"] = pd.to_datetime(sheldus_data["Hazard_start"])
    sheldus_data["Hazard_end"] = pd.to_datetime(sheldus_data["Hazard_end"])

    # Selecting the data within the desired date range (1980-2018)
    sheldus_data = sheldus_data[
        (sheldus_data["Hazard_start"].dt.year >= 1980)
        & (sheldus_data["Hazard_start"].dt.year <= 2018)
    ]

    # Extracting only "Hazard_start" and "Hazard_end" columns for the second DataFrame
    sheldus_dates = sheldus_data[["Hazard_start", "Hazard_end"]].copy()

    # Create a new column "ID" to uniquely identify each row in both DataFrames
    sheldus_dates["ID"] = range(1, len(sheldus_dates) + 1)
    sheldus_data["ID"] = sheldus_dates["ID"]

    return sheldus_data, sheldus_dates


# Function to load datasets for a county
def load_county_datasets(county_name, fips_code, SAVE_PATH, coast):
    """
    Load datasets for a given county based on its name and FIPS code.

    Parameters:
    - county_name (str): Name of the county.
    - fips_code (str): FIPS code of the county.
    - SAVE_PATH (str): The base directory path where all the results of the analysis will be saved.
    - coast (str): The coast to which the county belongs.

    Returns:
    - tuple: Datasets for precipitation, soil moisture, and river discharge for the county and datasets for storm surge and wave data.
    """
    # Constructing the paths for the input and output files specific to the current county
    base_path = f"{SAVE_PATH}{county_name}_{fips_code}/"
    PRECIPITATION_PATH = (
        f"../data/ready-for-analysis/precipitation/{county_name}_{fips_code}/"
    )
    SOIL_MOISTURE_PATH = (
        f"../data/ready-for-analysis/soil-moisture/{county_name}_{fips_code}/"
    )
    RIVER_DISCHARGE_PATH = (
        f"../data/ready-for-analysis/river-discharge/{county_name}_{fips_code}/"
    )
    WAVE_DATA_PATH = f"../data/ready-for-analysis/waves/combined_percentiles_with_wave_data_{coast}.csv"
    STORM_SURGE_PATH = (
        f"../data/ready-for-analysis/storm-surge/combined_ts_surge-percentiles.csv"
    )
    WATER_LEVEL_PATH = (
        f"../data/ready-for-analysis/storm-surge/combined_ts_waterlevel-percentiles.csv"
    )

    # Paths for the datasets
    precip_path = f"{PRECIPITATION_PATH}combined_ts_with_percentiles_{county_name}_{fips_code}.csv"
    soil_moisture_path = f"{SOIL_MOISTURE_PATH}combined_ts_with_percentiles_{county_name}_{fips_code}.csv"
    river_discharge_path = f"{RIVER_DISCHARGE_PATH}combined_ts_with_percentiles_{county_name}_{fips_code}.csv"

    # Loading the datasets
    precip_df = pd.read_csv(precip_path, parse_dates=["Date"])
    soil_moisture_df = pd.read_csv(soil_moisture_path, parse_dates=["Date"])
    river_discharge_df = pd.read_csv(river_discharge_path, parse_dates=["Date"])
    surge_data = pd.read_csv(STORM_SURGE_PATH, parse_dates=["Date_surge"])
    water_level_data = pd.read_csv(WATER_LEVEL_PATH, parse_dates=["Date_waterlevel"])
    wave_data = pd.read_csv(WAVE_DATA_PATH, parse_dates=["Date"])

    return (
        precip_df,
        soil_moisture_df,
        river_discharge_df,
        surge_data,
        water_level_data,
        wave_data,
    )


# Check the locations of all variables for a given county and FIPS code
def plot_locations(
    county_name,
    fips_code,
    precip_df,
    soil_moisture_df,
    discharge_df,
    surge_df,
    wave_df,
    base_path,
):
    """
    Plot the locations of all variables for a given county and FIPS code.

    Parameters:
    - county_name (str): Name of the county.
    - fips_code (str): FIPS code of the county.
    - precip_df (DataFrame): Precipitation dataset.
    - soil_moisture_df (DataFrame): Soil moisture dataset.
    - discharge_df (DataFrame): River discharge dataset.
    - surge_df (DataFrame): Storm surge dataset.
    - wave_df (DataFrame): Wave dataset.
    - base_path (str): The base directory path where all the results of the analysis will be saved.

    Returns:
    - None
    """
    # Map of all locations
    # stamen_terrain = cimgt.Stamen("terrain-background")
    osm_tiles = cimgt.OSM()
    fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(projection=osm_tiles.crs)
    # set extent of the map little bit more than the County boundary
    ax.set_extent(
        [
            precip_df["longitude"].min() - 0.5,
            precip_df["longitude"].max() + 0.5,
            precip_df["latitude"].min() - 0.5,
            precip_df["latitude"].max() + 0.5,
        ],
        crs=ccrs.Geodetic(),
    )
    # ax.set_extent([-125.9, -64.4, 23.0, 51.3], crs=ccrs.Geodetic()) # for US
    # Add the Stamen data at zoom level 8.
    ax.add_image(osm_tiles, 8)
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
        precip_df["longitude"],
        precip_df["latitude"],
        color="blue",
        s=35,
        transform=ccrs.Geodetic(),
        label="Precipitation",
    )
    ax.scatter(
        discharge_df["longitude"],
        discharge_df["latitude"],
        facecolors="none",
        edgecolors="r",
        linewidths=1,
        s=90,
        transform=ccrs.Geodetic(),
        label="River Discharge",
    )
    ax.scatter(
        soil_moisture_df["longitude"],
        soil_moisture_df["latitude"],
        facecolors="none",
        edgecolors="darkgreen",
        linewidths=1,
        s=150,
        transform=ccrs.Geodetic(),
        label="Soil Moisture",
    )
    ax.scatter(
        surge_df["longitude_surge"],
        surge_df["latitude_surge"],
        color="black",
        s=35,
        alpha=0.5,
        transform=ccrs.Geodetic(),
        label="Storm Surge",
    )
    ax.scatter(
        wave_df["longitude"],
        wave_df["latitude"],
        facecolors="none",
        edgecolors="orange",
        linewidths=1,
        # alpha=0.5,
        s=40,
        transform=ccrs.Geodetic(),
        label="Wave",
    )

    ax.set_title(
        f"Locations of Flood Drivers Data in {county_name}-{fips_code}",
        fontsize=18,
        y=1.03,
    )
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    legend = plt.legend(
        loc="upper center",
        fontsize=14,
        frameon=True,
    )  # bbox_to_anchor=(0.005, 0.4), loc='upper left', borderaxespad=1,

    # Set the legend face color to white
    legend.get_frame().set_facecolor("white")

    # Ensure the directories exist
    ensure_county_directories_exist(county_name, fips_code)

    plt.savefig(
        f"{base_path}plots/locations of all flood vars data_{county_name}_{fips_code}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.close()


# Set dates as index for all datasets and change them to datetime format
def set_dates_as_index(
    precip_df, soil_moisture_df, discharge_df, surge_df, water_level_data, wave_df
):
    """
    Set dates as index for all datasets and change them to datetime format.

    Parameters:
    - precip_df (DataFrame): Precipitation dataset.
    - soil_moisture_df (DataFrame): Soil moisture dataset.
    - discharge_df (DataFrame): River discharge dataset.
    - surge_df (DataFrame): Storm surge dataset.
    - water_level_data (DataFrame): Water level dataset.
    - wave_df (DataFrame): Wave dataset.

    Returns:
    - tuple: Precipitation, soil moisture, river discharge, storm surge, water level, and wave datasets with dates as index.
    """
    # Set dates as index for all datasets and change them to datetime format
    precip_df = precip_df.set_index("Date")
    precip_df.index = pd.to_datetime(precip_df.index)

    soil_moisture_df = soil_moisture_df.set_index("Date")
    soil_moisture_df.index = pd.to_datetime(soil_moisture_df.index)

    discharge_df = discharge_df.set_index("Date")
    discharge_df.index = pd.to_datetime(discharge_df.index)

    surge_df["Date"] = surge_df["Date_surge"]
    surge_df = surge_df.set_index("Date")
    surge_df.index = pd.to_datetime(surge_df.index)

    water_level_data["Date"] = water_level_data["Date_waterlevel"]
    water_level_data = water_level_data.set_index("Date")
    water_level_data.index = pd.to_datetime(water_level_data.index)

    # Rename waterlevel_percentile to waterlevel_percentiles
    water_level_data = water_level_data.rename(
        columns={
            "waterlevel_percentile": "waterlevel_percentiles",
        }
    )

    # Change date to datetime format
    wave_df["Date"] = pd.to_datetime(wave_df["Date"])

    # Remove time from the date column
    wave_df["Date"] = wave_df["Date"].dt.date
    wave_df = wave_df.set_index("Date")
    wave_df.index = pd.to_datetime(wave_df.index)

    return (
        precip_df,
        soil_moisture_df,
        discharge_df,
        surge_df,
        water_level_data,
        wave_df,
    )


# Rename the `Percentiles` and `latitude` & `longitude` columns and create a new column for the Date_{variable}
def rename_columns(
    precip_df, soil_moisture_df, discharge_df, wave_df, base_path, county_name, FIPS
):
    """
    Rename the `Percentiles` and `latitude` & `longitude` columns of the datasets to make them unique
    to each dataset and create a new column for the Date_{variable} for each dataset.

    Parameters:
    - precip_df (DataFrame): Precipitation dataset.
    - soil_moisture_df (DataFrame): Soil moisture dataset.
    - discharge_df (DataFrame): River discharge dataset.
    - wave_df (DataFrame): Wave dataset.
    - base_path (str): The base directory path where all the results of the analysis will be saved.
    - county_name (str): Name of the county.
    - FIPS (str): FIPS code of the county.

    Returns:
    - tuple: Precipitation, soil moisture, river discharge, and wave datasets with renamed columns.

    """

    # Rename the `Percentiles` and `latitude` & `longitude` columns of the datasets
    precip_df = precip_df.rename(
        columns={
            "Percentiles": "Percentiles_precip",
            "latitude": "latitude_precip",
            "longitude": "longitude_precip",
        }
    )
    precip_df["Date_precip"] = precip_df.index
    precip_df["Date_precip"] = pd.to_datetime(precip_df["Date_precip"])

    # save the data for precipitation
    precip_df.to_csv(
        f"{base_path}data/precipitation_{county_name}_{FIPS}.csv", index=False
    )

    soil_moisture_df = soil_moisture_df.rename(
        columns={
            "Percentiles": "Percentiles_soil_moisture",
            "latitude": "latitude_soil_moisture",
            "longitude": "longitude_soil_moisture",
        }
    )
    soil_moisture_df["Date_soil_moisture"] = soil_moisture_df.index
    soil_moisture_df["Date_soil_moisture"] = pd.to_datetime(
        soil_moisture_df["Date_soil_moisture"]
    )

    # save the data for soil moisture
    soil_moisture_df.to_csv(
        f"{base_path}data/soil-moisture_{county_name}_{FIPS}.csv", index=False
    )

    discharge_df = discharge_df.rename(
        columns={
            "Percentiles": "Percentiles_discharge",
            "latitude": "latitude_discharge",
            "longitude": "longitude_discharge",
        }
    )
    discharge_df["Date_discharge"] = discharge_df.index
    discharge_df["Date_discharge"] = pd.to_datetime(discharge_df["Date_discharge"])
    # save the data for river discharge
    discharge_df.to_csv(
        f"{base_path}data/river-discharge_{county_name}_{FIPS}.csv", index=False
    )

    wave_df = wave_df.rename(
        columns={
            "latitude": "latitude_wave",
            "longitude": "longitude_wave",
        }
    )
    wave_df["Date_wave"] = wave_df.index
    wave_df["Date_wave"] = pd.to_datetime(wave_df["Date_wave"])

    return precip_df, soil_moisture_df, discharge_df, wave_df


# Read the dataset for precipitation unique locations for a given county and FIPS code
def read_precipitation_locations_data(county_name, FIPS):
    """
    Read the precipitation locations dataset for a given county.

    Parameters:
    - county_name (str): Name of the county.
    - FIPS (str): FIPS code of the county.

    Returns:
    - DataFrame: Precipitation data locations for a given county and FIPS code.
    """
    # Path to the precipitation locations dataset
    PRECIPITATION_PATH = (
        f"../data/ready-for-analysis/precipitation/{county_name}_{FIPS}/"
    )

    # Read the dataset for river discharge unique locations for a given county and FIPS code
    precip_locations_df = pd.read_csv(
        f"{PRECIPITATION_PATH}unique_locations/{county_name}_unique_locations.csv",
    )

    return precip_locations_df


# Read the coastline data
def read_coastline_data():
    """
    Read the coastline data. The data is filtered to only include the Gulf of Mexico and Atlantic coasts. The columns
    are renamed to match the other datasets.

    Returns:
    - DataFrame: Coastline data.
    """
    # Path to the coastline data
    COASTLINE_PATH = "../data/coastline/filtered_coastline_gom_atlantic_coasts.csv"

    # Read the coastline data
    coastline_df = pd.read_csv(f"{COASTLINE_PATH}")

    # Rename the columns
    coastline_df = coastline_df.rename(
        columns={"Longitude": "longitude_coastline", "Latitude": "latitude_coastline"}
    )

    return coastline_df


# Combine the datasets for all variables for a given county and FIPS code based on the dates and locations
def combine_datasets(
    precip_df,
    soil_moisture_df,
    discharge_df,
):
    """
    Combine the datasets for all variables for a given county and FIPS code based on their dates and locations,
    using soil moisture data from one day before as a precondition, while keeping the first day's data.

    Parameters:
    - precip_df (DataFrame): Precipitation dataset.
    - soil_moisture_df (DataFrame): Soil moisture dataset.
    - discharge_df (DataFrame): River discharge dataset.


    Returns:
    - DataFrame: Combined dataset for all variables.
    """

    # Print that  the datasets are being merged
    print("Merging the datasets for all variables...")

    # Change date columns to datetime format for comparison
    precip_df["Date_precip"] = pd.to_datetime(precip_df["Date_precip"])
    soil_moisture_df["Date_soil_moisture"] = pd.to_datetime(
        soil_moisture_df["Date_soil_moisture"]
    )
    discharge_df["Date_discharge"] = pd.to_datetime(discharge_df["Date_discharge"])

    # Create a copy of soil moisture data for the next day
    soil_moisture_next_df = soil_moisture_df.copy()
    soil_moisture_next_df["Date_soil_moisture"] = soil_moisture_next_df[
        "Date_soil_moisture"
    ] + pd.Timedelta(days=1)

    # Combine the original and next-day soil moisture data
    soil_moisture_combined = pd.concat(
        [soil_moisture_df, soil_moisture_next_df]
    ).drop_duplicates(
        subset=[
            "Date_soil_moisture",
            "latitude_soil_moisture",
            "longitude_soil_moisture",
        ],
        keep="first",
    )

    # Limit latitude and longitude to 4 decimal places for comparison
    precip_df["latitude_precip"] = precip_df["latitude_precip"].round(4)
    precip_df["longitude_precip"] = precip_df["longitude_precip"].round(4)
    soil_moisture_combined["latitude_soil_moisture"] = soil_moisture_combined[
        "latitude_soil_moisture"
    ].round(4)
    soil_moisture_combined["longitude_soil_moisture"] = soil_moisture_combined[
        "longitude_soil_moisture"
    ].round(4)
    discharge_df["latitude_discharge"] = discharge_df["latitude_discharge"].round(4)
    discharge_df["longitude_discharge"] = discharge_df["longitude_discharge"].round(4)

    # Merge precipitation and soil moisture datasets
    combined_vars_df = pd.merge(
        precip_df,
        soil_moisture_combined,
        how="left",
        left_on=["Date_precip", "latitude_precip", "longitude_precip"],
        right_on=[
            "Date_soil_moisture",
            "latitude_soil_moisture",
            "longitude_soil_moisture",
        ],
    )

    # Merge the result with the discharge dataset
    combined_vars_df = pd.merge(
        combined_vars_df,
        discharge_df,
        how="left",
        left_on=["Date_precip", "latitude_precip", "longitude_precip"],
        right_on=["Date_discharge", "latitude_discharge", "longitude_discharge"],
    )

    return combined_vars_df


def calculate_distance_and_assign_points(coastline_df, combined_vars_df):
    """
    Calculate the distance between each location in combined_vars_df and the nearest coastline point in coastline_df
    within a bounding box defined by the county. Assign 'Yes' or 'No' to 'surge_points' and 'wave_points' in
    combined_vars_df based on the distance criteria.

    Args:
        coastline_df (pd.DataFrame): DataFrame containing coastline points.
        combined_vars_df (pd.DataFrame): DataFrame containing combined variables data.

    Returns:
        pd.DataFrame: Updated combined_vars_df with 'surge_points' and 'wave_points' columns.
    """

    # Print that the surge and wave points are being assigned
    print(
        "Calculating the distance to the nearest coastline point and assigning surge and wave points..."
    )

    surge_threshold_km = 6  # Threshold for surge points in kilometers
    wave_threshold_km = 6  # Threshold for wave points in kilometers
    buffer = 0.5  # Buffer in degrees to include a little more area

    # Initialize columns for distances and points
    combined_vars_df["surge_points"] = "No"
    combined_vars_df["wave_points"] = "No"

    # Calculate bounding box for the county
    min_lon, max_lon = (
        combined_vars_df["longitude_precip"].min() - buffer,
        combined_vars_df["longitude_precip"].max() + buffer,
    )
    min_lat, max_lat = (
        combined_vars_df["latitude_precip"].min() - buffer,
        combined_vars_df["latitude_precip"].max() + buffer,
    )

    # Filter coastline points within the bounding box
    coastline_filtered = coastline_df[
        (coastline_df["longitude_coastline"] >= min_lon)
        & (coastline_df["longitude_coastline"] <= max_lon)
        & (coastline_df["latitude_coastline"] >= min_lat)
        & (coastline_df["latitude_coastline"] <= max_lat)
    ]

    # Convert filtered coast points and location points to tuples (latitude, longitude)
    coast_points = list(
        zip(
            coastline_filtered["latitude_coastline"],
            coastline_filtered["longitude_coastline"],
        )
    )
    location_points = list(
        zip(combined_vars_df["latitude_precip"], combined_vars_df["longitude_precip"])
    )

    for i, loc_point in tqdm(enumerate(location_points)):
        # Calculate distances from this location to filtered coast points
        distances = [
            haversine(loc_point, coast_point, unit=Unit.KILOMETERS)
            for coast_point in coast_points
        ]

        # Find the minimum distance to the coast
        min_distance_to_coast = min(distances)

        # Assign 'Yes' or 'No' based on distance criteria
        if min_distance_to_coast <= surge_threshold_km:
            combined_vars_df.at[i, "surge_points"] = "Yes"

        if min_distance_to_coast <= wave_threshold_km:
            combined_vars_df.at[i, "wave_points"] = "Yes"

    return combined_vars_df


def find_nearest_data(
    combined_vars_df,
    surge_df,
    wave_df,
    waterlevel_df,
    is_gulf_coast,
    sheldus_df,
    window_size=3,
):
    """
    Find the nearest surge, wave, and waterlevel data points for each date in combined_vars_df.
    This function adds detailed information about the nearest locations to combined_vars_df,
    including latitude, longitude, date, and their respective percentiles.

    Parameters:
    combined_vars_df (pd.DataFrame): The dataframe containing various points for each date with latitude and longitude information.
    surge_df (pd.DataFrame): Dataframe containing storm surge data with dates, latitudes, longitudes, and percentiles.
    wave_df (pd.DataFrame): Dataframe containing wave data with dates, latitudes, longitudes, and percentiles.
    waterlevel_df (pd.DataFrame): Dataframe containing water level data with dates, latitudes, longitudes, and percentiles.
    sheldus_df (pd.DataFrame): Dataframe containing hazard start and end dates.
    coast (str): Indicates whether the coast is 'GulfOfMexico' or 'Atlantic'.
    window_size (int): The number of days to expand the date window for filtering.

    Returns:
    pd.DataFrame: The modified combined_vars_df with detailed information about the nearest surge, wave, and waterlevel data.
    """

    # Convert date columns to datetime format for comparison
    combined_vars_df["Date_precip"] = pd.to_datetime(combined_vars_df["Date_precip"])
    surge_df["Date_surge"] = pd.to_datetime(surge_df["Date_surge"])
    wave_df["Date_wave"] = pd.to_datetime(wave_df["Date_wave"])

    waterlevel_df["Date_waterlevel"] = pd.to_datetime(waterlevel_df["Date_waterlevel"])

    # Limit latitude and longitude to 4 decimal places for consistent comparison
    decimal_places = 4
    combined_vars_df["latitude_precip"] = combined_vars_df["latitude_precip"].round(
        decimal_places
    )
    combined_vars_df["longitude_precip"] = combined_vars_df["longitude_precip"].round(
        decimal_places
    )

    surge_df["latitude_surge"] = surge_df["latitude_surge"].round(decimal_places)
    surge_df["longitude_surge"] = surge_df["longitude_surge"].round(decimal_places)

    wave_df["latitude_wave"] = wave_df["latitude_wave"].round(decimal_places)
    wave_df["longitude_wave"] = wave_df["longitude_wave"].round(decimal_places)

    waterlevel_df["latitude_waterlevel"] = waterlevel_df["latitude_waterlevel"].round(
        decimal_places
    )
    waterlevel_df["longitude_waterlevel"] = waterlevel_df["longitude_waterlevel"].round(
        decimal_places
    )

    # Subset the dataframes based on the SHELDUS hazard start and end dates
    output_combined_vars_df = pd.DataFrame()
    output_surge_df = pd.DataFrame()
    output_wave_df = pd.DataFrame()
    output_waterlevel_df = pd.DataFrame()

    for i in range(len(sheldus_df)):
        # Get the start and end dates for the hazard event and the window dates
        start_date = sheldus_df["Hazard_start"].iloc[i]
        end_date = sheldus_df["Hazard_end"].iloc[i]
        first_window_date = start_date - timedelta(days=window_size)
        last_window_date = end_date + timedelta(days=window_size)
        window_dates = pd.date_range(first_window_date, last_window_date)

        # Subset the dataframes based on the window dates
        window_combined_vars = combined_vars_df.loc[
            combined_vars_df["Date_precip"].isin(window_dates)
        ]
        window_surge = surge_df.loc[surge_df["Date_surge"].isin(window_dates)]
        window_wave = wave_df.loc[wave_df["Date_wave"].isin(window_dates)]
        window_waterlevel = waterlevel_df.loc[
            waterlevel_df["Date_waterlevel"].isin(window_dates)
        ]

        # Concatenate the subsetted dataframes
        output_combined_vars_df = pd.concat(
            [output_combined_vars_df, window_combined_vars], axis=0
        )
        output_surge_df = pd.concat([output_surge_df, window_surge], axis=0)
        output_wave_df = pd.concat([output_wave_df, window_wave], axis=0)
        output_waterlevel_df = pd.concat(
            [output_waterlevel_df, window_waterlevel], axis=0
        )

    # Update the dataframes with the subsetted data
    combined_vars_df = output_combined_vars_df.copy()
    surge_df = output_surge_df.copy()
    wave_df = output_wave_df.copy()
    waterlevel_df = output_waterlevel_df.copy()

    # Reset the index of the combined_vars_df
    combined_vars_df.reset_index(drop=True, inplace=True)

    # Print the combined_vars_df
    print("Combined variables data for all sheldus events: ", combined_vars_df)
    print("Subsetted surge data: ", surge_df)
    print("Subsetted wave data: ", wave_df)
    print("Subsetted water level data: ", waterlevel_df)

    # Check if the county has all surge and wave points as "NO"
    if (combined_vars_df["surge_points"] == "NO").all() and (
        combined_vars_df["wave_points"] == "NO"
    ).all():
        print(
            "The county has all surge and wave points as 'NO'. Skipping nearest data calculation."
        )
        return combined_vars_df

    # Check if the county has surge_points as "Yes" but all wave_points as "NO"
    if (combined_vars_df["surge_points"] == "Yes").any() and (
        combined_vars_df["wave_points"] == "NO"
    ).all():
        print(
            "The county has surge_points as 'Yes' but all wave_points as 'NO'. Skipping nearest wave data calculation."
        )
        calculate_wave_distances = False
    else:
        calculate_wave_distances = True

    # Calculate bounding box for the county
    buffer = 2.0  # Buffer in degrees to include a little more area
    min_lon, max_lon = (
        combined_vars_df["longitude_precip"].min() - buffer,
        combined_vars_df["longitude_precip"].max() + buffer,
    )
    min_lat, max_lat = (
        combined_vars_df["latitude_precip"].min() - buffer,
        combined_vars_df["latitude_precip"].max() + buffer,
    )

    # Filter surge, wave, and water level data within the bounding box
    surge_filtered = surge_df[
        (surge_df["longitude_surge"] >= min_lon)
        & (surge_df["longitude_surge"] <= max_lon)
        & (surge_df["latitude_surge"] >= min_lat)
        & (surge_df["latitude_surge"] <= max_lat)
    ]
    wave_filtered = wave_df[
        (wave_df["longitude_wave"] >= min_lon)
        & (wave_df["longitude_wave"] <= max_lon)
        & (wave_df["latitude_wave"] >= min_lat)
        & (wave_df["latitude_wave"] <= max_lat)
    ]
    waterlevel_filtered = waterlevel_df[
        (waterlevel_df["longitude_waterlevel"] >= min_lon)
        & (waterlevel_df["longitude_waterlevel"] <= max_lon)
        & (waterlevel_df["latitude_waterlevel"] >= min_lat)
        & (waterlevel_df["latitude_waterlevel"] <= max_lat)
    ]

    # Print the filtered surge, wave, and water level data
    print("Filtered surge data: ", surge_filtered.head())
    print("Filtered wave data: ", wave_filtered.head())
    print("Filtered water level data: ", waterlevel_filtered.head())

    # Reset the index of the filtered dataframes
    surge_filtered.reset_index(drop=True, inplace=True)
    wave_filtered.reset_index(drop=True, inplace=True)
    waterlevel_filtered.reset_index(drop=True, inplace=True)

    # Print the filtered surge, wave, and water level data
    print("Filtered surge data with reset index: ", surge_filtered.head())
    print("Filtered wave data with reset index: ", wave_filtered.head())
    print("Filtered water level data with reset index: ", waterlevel_filtered.head())

    # Print info about the filtered dataframes
    print("Info about the filtered surge data: ", surge_filtered.info())
    print("Info about the filtered wave data: ", wave_filtered.info())
    print("Info about the filtered water level data: ", waterlevel_filtered.info())

    # Validate DataFrame shapes
    print("Surge data shape:", surge_filtered.shape)
    print("Wave data shape:", wave_filtered.shape)
    print("Water level data shape:", waterlevel_filtered.shape)

    # Change date columns to datetime format for comparison
    combined_vars_df["Date_precip"] = pd.to_datetime(combined_vars_df["Date_precip"])
    surge_filtered["Date_surge"] = pd.to_datetime(surge_filtered["Date_surge"])
    wave_filtered["Date_wave"] = pd.to_datetime(wave_filtered["Date_wave"])
    waterlevel_filtered["Date_waterlevel"] = pd.to_datetime(
        waterlevel_filtered["Date_waterlevel"]
    )

    # Create new columns to store the nearest surge, wave, and waterlevel data
    combined_vars_df["Date_surge"] = np.nan
    combined_vars_df["latitude_surge"] = np.nan
    combined_vars_df["longitude_surge"] = np.nan
    combined_vars_df["surge_percentiles"] = np.nan
    combined_vars_df["distance_surge"] = np.nan  # New distance column

    combined_vars_df["Date_wave"] = np.nan
    combined_vars_df["latitude_wave"] = np.nan
    combined_vars_df["longitude_wave"] = np.nan
    combined_vars_df["waveHs_percentiles"] = np.nan
    combined_vars_df["distance_wave"] = np.nan  # New distance column

    combined_vars_df["Date_waterlevel"] = np.nan
    combined_vars_df["latitude_waterlevel"] = np.nan
    combined_vars_df["longitude_waterlevel"] = np.nan
    combined_vars_df["waterlevel_percentiles"] = np.nan
    combined_vars_df["distance_waterlevel"] = np.nan  # New distance column

    # Ensure 'distances' is initialized
    distances = None  # Initialize to avoid referencing before assignment

    # Iterate through each row in the combined_vars_df
    for index, row in combined_vars_df.iterrows():
        date = row["Date_precip"]

        # Coordinates of the current point based on coast
        if is_gulf_coast:
            precip_coord = np.array([row["latitude_precip"], row["longitude_precip"]])
        else:
            precip_coord = np.array([row["longitude_precip"], row["latitude_precip"]])

        # Check if 'surge_points' is 'Yes'
        if row["surge_points"] == "Yes":
            relevant_surge = surge_filtered[surge_filtered["Date_surge"] == date]
            if not relevant_surge.empty:
                # Get the coordinates of all surge points based on coast
                if is_gulf_coast:
                    distances = haversine_vector(
                        [precip_coord],
                        relevant_surge[
                            ["latitude_surge", "longitude_surge"]
                        ].to_numpy(),
                        Unit.KILOMETERS,
                        comb=True,
                    )

                else:

                    distances = haversine_vector(
                        [precip_coord],
                        relevant_surge[
                            ["longitude_surge", "latitude_surge"]
                        ].to_numpy(),
                        Unit.KILOMETERS,
                        comb=True,
                    )

                nearest_surge_index = np.argmin(distances)
                nearest_surge_data = relevant_surge.iloc[nearest_surge_index]

                # Populate the new columns with the nearest surge data
                combined_vars_df.at[index, "Date_surge"] = nearest_surge_data[
                    "Date_surge"
                ]
                combined_vars_df.at[index, "latitude_surge"] = nearest_surge_data[
                    "latitude_surge"
                ]
                combined_vars_df.at[index, "longitude_surge"] = nearest_surge_data[
                    "longitude_surge"
                ]
                combined_vars_df.at[index, "surge_percentiles"] = nearest_surge_data[
                    "surge_percentiles"
                ]
                combined_vars_df.at[index, "distance_surge"] = distances[
                    nearest_surge_index
                ]  # Store distance

        # Check if 'wave_points' is 'Yes'
        if row["wave_points"] == "Yes" and calculate_wave_distances == True:
            if is_gulf_coast:
                relevant_wave = wave_df[wave_df["Date_wave"] == date]
            else:
                relevant_wave = wave_df[wave_df["Date_wave"] == date]

            if not relevant_wave.empty:
                if is_gulf_coast:

                    distances = haversine_vector(
                        [precip_coord],
                        relevant_wave[["latitude_wave", "longitude_wave"]].to_numpy(),
                        Unit.KILOMETERS,
                        comb=True,
                    )
                else:

                    distances = haversine_vector(
                        [precip_coord],
                        relevant_wave[["longitude_wave", "latitude_wave"]].to_numpy(),
                        Unit.KILOMETERS,
                        comb=True,
                    )

                nearest_wave_index = np.argmin(distances)
                nearest_wave_data = relevant_wave.iloc[nearest_wave_index]

                # Populate the new columns with the nearest wave data
                combined_vars_df.at[index, "Date_wave"] = nearest_wave_data["Date_wave"]
                combined_vars_df.at[index, "latitude_wave"] = nearest_wave_data[
                    "latitude_wave"
                ]
                combined_vars_df.at[index, "longitude_wave"] = nearest_wave_data[
                    "longitude_wave"
                ]
                combined_vars_df.at[index, "waveHs_percentiles"] = nearest_wave_data[
                    "waveHs_percentiles"
                ]
                combined_vars_df.at[index, "distance_wave"] = distances[
                    nearest_wave_index
                ]  # Store distance

        # Check if 'surge_points' is 'Yes'
        if row["surge_points"] == "Yes":
            # Find the nearest waterlevel point for the given date
            relevant_waterlevel = waterlevel_filtered[
                waterlevel_filtered["Date_waterlevel"] == date
            ]

            if not relevant_waterlevel.empty:
                if is_gulf_coast:

                    distances = haversine_vector(
                        [precip_coord],
                        relevant_waterlevel[
                            ["latitude_waterlevel", "longitude_waterlevel"]
                        ].to_numpy(),
                        Unit.KILOMETERS,
                        comb=True,
                    )
                else:

                    distances = haversine_vector(
                        [precip_coord],
                        relevant_waterlevel[
                            ["longitude_waterlevel", "latitude_waterlevel"]
                        ].to_numpy(),
                        Unit.KILOMETERS,
                        comb=True,
                    )

                nearest_waterlevel_index = np.argmin(distances)
                nearest_waterlevel_data = relevant_waterlevel.iloc[
                    nearest_waterlevel_index
                ]

                # Populate the new columns with the nearest waterlevel data
                combined_vars_df.at[index, "Date_waterlevel"] = nearest_waterlevel_data[
                    "Date_waterlevel"
                ]
                combined_vars_df.at[index, "latitude_waterlevel"] = (
                    nearest_waterlevel_data["latitude_waterlevel"]
                )
                combined_vars_df.at[index, "longitude_waterlevel"] = (
                    nearest_waterlevel_data["longitude_waterlevel"]
                )
                combined_vars_df.at[index, "waterlevel_percentiles"] = (
                    nearest_waterlevel_data["waterlevel_percentiles"]
                )
                combined_vars_df.at[index, "distance_waterlevel"] = distances[
                    nearest_waterlevel_index
                ]  # Store distance

    return combined_vars_df


def calculate_average_percentile(combined_vars_df):
    """
    Calculate the average percentile of all the drivers for each county based on the conditions of 'surge_points' and 'wave_points'.
    Create a new column called "average_percentile" in the combined dataframe and fill it with the calculated values.

    Parameters:
    -----------
    combined_vars_df : pd.DataFrame
        DataFrame containing combined variables data with 'surge_points', 'wave_points', 'Percentiles_precip',
        'Percentiles_discharge', 'Percentiles_soil_moisture', 'nearest_surge', and 'nearest_wave' columns.

    Returns:
    --------
    pd.DataFrame
        The updated combined_vars_df with a new column 'average_percentile' containing the calculated average percentile values.
    """

    # Print that the average percentile is being calculated
    print("Calculating the average percentile...")

    def calculate_percentile(row):
        """
        Helper function to calculate the average percentile based on the conditions of 'surge_points' and 'wave_points'.

        Parameters:
        -----------
        row : pd.Series
            A single row of the combined_vars_df DataFrame.

        Returns:
        --------
        float or None
            The calculated average percentile value, or None if all percentiles are None.
        """

        # Check if the location point is a surge point and a wave point
        if row["surge_points"] == "Yes" and row["wave_points"] == "Yes":
            # Add the percentiles of precipitation, river discharge, soil moisture, storm surge, and waves, and divide by 5
            percentiles = [
                row["Percentiles_precip"],
                row["Percentiles_discharge"],
                row["Percentiles_soil_moisture"],
                row["surge_percentiles"],
                row["waveHs_percentiles"],
            ]
        # Check if the location point is a surge point but not a wave point
        elif row["surge_points"] == "Yes" and row["wave_points"] == "No":
            # Add the percentiles of precipitation, river discharge, soil moisture, and storm surge, and divide by 4
            percentiles = [
                row["Percentiles_precip"],
                row["Percentiles_discharge"],
                row["Percentiles_soil_moisture"],
                row["surge_percentiles"],
            ]
        # If the location point is not a surge point and not a wave point and there is no discharge data available, skip that point and return None
        elif (
            row["surge_points"] == "No"
            and row["wave_points"] == "No"
            and pd.isnull(row["Percentiles_discharge"])
        ):
            return None

        # If the location point is not a surge point and not a wave point
        else:
            # Add the percentiles of precipitation, river discharge, and soil moisture, and divide by 3
            percentiles = [
                row["Percentiles_precip"],
                row["Percentiles_discharge"],
                row["Percentiles_soil_moisture"],
            ]

        # Filter out None values from the percentiles list
        valid_percentiles = [p for p in percentiles if p is not None]

        # If there are valid percentiles, calculate the average
        if valid_percentiles:
            return sum(valid_percentiles) / len(valid_percentiles)
        else:
            return None

    # Apply the calculate_percentile function to each row of the combined_vars_df DataFrame
    combined_vars_df["average_percentile"] = combined_vars_df.apply(
        calculate_percentile, axis=1
    )

    return combined_vars_df


########################################################################################################
#################################### PROCESSING COUNTIES FUNCTION ######################################
########################################################################################################


def process_county(
    row,
    SHELDUS_COUNTIES_PATH,
    SHELDUS_SAVE_PATH,
    STORM_SURGE_PATH,
    WATER_LEVEL_PATH,
    WAVE_PATH,
    SAVE_PATH,
):
    """
    This function performs a comprehensive flood impact analysis for a different counties based on various climate
    and geographical datasets. The analysis involves data loading, preprocessing, analysis, and saving the results.

    Parameters:
    row (pd.Series): A row from the DataFrame containing county information. It must include 'County', 'FIPS', 'State', and 'Coast' columns.
    SHELDUS_COUNTIES_PATH (str): The directory path where the SHELDUS data for all counties is stored. Each county's data should be in a separate file.
    SHELDUS_SAVE_PATH (str): The directory path where the processed SHELDUS data will be saved.
    STORM_SURGE_PATH (str): The file path for the storm surge data.
    WATER_LEVEL_PATH (str): The file path for the water level data.
    WAVE_PATH (str): The directory path where the wave data is stored.
    SAVE_PATH (str): The base directory path where all the results of the analysis will be saved. Each county's results will be saved in a separate subdirectory.

    Returns:
    None: The function doesn't return anything. Instead, it saves the analysis results to files.

    Raises:
    FileNotFoundError: If any of the required data files or directories are not found.
    Exception: For any other issues that might occur during the processing.
    """

    # Path to the directory where the precipitation data is stored
    precipitation_data_path = "../data/ready-for-analysis/precipitation/"

    # Create a DataFrame with the county names, FIPS codes, state abbreviations and the coast they belong to.
    county_names_fips_codes_states_coasts = (
        create_county_names_fips_codes_states_coasts_df(
            SHELDUS_COUNTIES_PATH, SHELDUS_SAVE_PATH, precipitation_data_path
        )
    )

    try:
        # Extracting county information from the input row
        county_name = row["County"]
        FIPS = row["FIPS"]
        state = row["State"]
        coast = row["Coast"]

        # Constructing the paths for the input and output files specific to the current county
        base_path = f"{SAVE_PATH}{county_name}_{FIPS}/"

        # Load datasets for a given county and FIPS code
        (
            precip_df,
            soil_moisture_df,
            river_discharge_df,
            surge_data,
            water_level_data,
            wave_df,
        ) = load_county_datasets(county_name, FIPS, SAVE_PATH, coast)

        # Select the wave data for the county and FIPS code
        # wave_df = select_wave_data(county_name, FIPS, county_names_fips_codes_states_coasts, WAVE_PATH)

        # Print a line break for better readability
        print("\n")

        print(f"{coast} Coast")

        # Check the data for a given county and FIPS code
        print(f"Precipitation data for {county_name} and {FIPS} code:")
        print(precip_df)
        print(f"Soil moisture data for {county_name} and {FIPS} code:")
        print(soil_moisture_df)
        print(f"River discharge data for {county_name} and {FIPS} code:")
        print(river_discharge_df)
        print(f"Storm surge data for {county_name} and {FIPS} code:")
        print(surge_data)
        print(f"Water level data for {county_name} and {FIPS} code:")
        print(water_level_data)
        print(f"Wave data for {county_name} and {FIPS} code on {coast}:")
        print(wave_df)

        # Plot the locations of all variables for a given county and FIPS code
        plot_locations(
            county_name,
            FIPS,
            precip_df,
            soil_moisture_df,
            river_discharge_df,
            surge_data,
            wave_df,
            base_path,
        )

        # Use the function to read data for a specific county and FIPS code
        try:
            sheldus_county_data, sheldus_events_dates = read_sheldus_data(
                county_name, FIPS, SHELDUS_COUNTIES_PATH
            )
            print("SHELDUS Data loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except FileExistsError as e:
            print(f"Error: {str(e)}")

        # Save the SHELDUS data for a given county and FIPS code
        sheldus_county_data.to_csv(
            f"{base_path}data/sheldus_data_{county_name}_{FIPS}.csv", index=False
        )

        # Save the SHELDUS dates for a given county and FIPS code
        sheldus_events_dates.to_csv(
            f"{base_path}data/sheldus_dates_{county_name}_{FIPS}.csv", index=False
        )

        # Check the SHELDUS data for a given county and FIPS code
        print(f"SHELDUS data for {county_name} and {FIPS} code on {coast}:")
        print(sheldus_county_data)

        # Check the SHELDUS dates for a given county and FIPS code
        print(f"SHELDUS dates for {county_name} and {FIPS} code:")
        print(sheldus_events_dates)

        # Set dates as index for all datasets and change them to datetime format
        (
            precip_df,
            soil_moisture_df,
            river_discharge_df,
            surge_df,
            water_level_df,
            wave_df,
        ) = set_dates_as_index(
            precip_df,
            soil_moisture_df,
            river_discharge_df,
            surge_data,
            water_level_data,
            wave_df,
        )

        # Rename the `Percentiles` and `latitude` & `longitude` columns of the datasets
        precip_df, soil_moisture_df, discharge_df, wave_df = rename_columns(
            precip_df,
            soil_moisture_df,
            river_discharge_df,
            wave_df,
            base_path,
            county_name,
            FIPS,
        )

        # Check the data for a given county and FIPS code
        print(
            f"Precipitation data for {county_name} with FIPS code {FIPS} after renaming:"
        )
        print(precip_df)
        print(
            f"Soil moisture data for {county_name} with FIPS code {FIPS} after renaming:"
        )
        print(soil_moisture_df)
        print(
            f"River discharge data for {county_name} with FIPS code {FIPS} after renaming:"
        )
        print(discharge_df)
        print(
            f"Storm surge data for {county_name} with FIPS code {FIPS} after renaming:"
        )
        print(surge_df)
        print(
            f"Water level data for {county_name} with FIPS code {FIPS} after renaming:"
        )
        print(water_level_df)
        print(
            f"Wave data for {county_name} with FIPS code {FIPS} on {coast} after renaming:"
        )
        print(wave_df)

        # Read the dataset for precipitation unique locations for a given county and FIPS code
        precip_locations_df = read_precipitation_locations_data(county_name, FIPS)

        print("Precipitation locations data:")
        print(precip_locations_df.head())

        # Read the coastline data
        coastline_df = read_coastline_data()

        print("Coastline data:")
        print(coastline_df.head())

        # Combine the datasets for all variables for a given county and FIPS code based on the dates and locations
        combined_vars_df = combine_datasets(
            precip_df,
            soil_moisture_df,
            discharge_df,
        )

        # Save the combined_vars_df
        combined_vars_df.to_csv(
            f"{base_path}data/combined_vars_{county_name}_{FIPS}.csv", index=False
        )

        print("Combined variables data:")
        print(combined_vars_df)

        # Calculate the distance between each location and the nearest coastline point in coastline_df.
        # Assign 'Yes' or 'No' to 'surge_points' and 'wave_points' in combined_vars_df based on the distance criteria.
        combined_vars_df = calculate_distance_and_assign_points(
            coastline_df, combined_vars_df
        )

        # Save the updated combined_vars_df
        combined_vars_df.to_csv(
            f"{base_path}data/combined_vars_with_surge_wave_points_{county_name}_{FIPS}.csv",
            index=False,
        )

        print(
            f"Combined variables data with surge and wave points for {county_name} on {coast}:"
        )
        print(combined_vars_df)

        # Determine if the county is on the Gulf of Mexico coast
        is_gulf_coast = coast == "GulfOfMexico"

        # Find the nearest percentile values of storm surge, wave, and water level based on the conditions of 'surge_points' and 'wave_points'
        combined_vars_df = find_nearest_data(
            combined_vars_df,
            surge_df,
            wave_df,
            water_level_df,
            is_gulf_coast,
            sheldus_events_dates,
        )

        # Save the updated combined_vars_df
        combined_vars_df.to_csv(
            f"{base_path}data/combined_vars_with_nearest_percentiles_{county_name}_{FIPS}.csv",
            index=False,
        )

        # Also save the combined_vars_df with the nearest data for the surge, wave, and water level to directory "E:/storm-surge-flooding-analysis/data/counties_combined_dfs/", instead od base_path
        combined_vars_df.to_csv(
            f"E:/storm-surge-flooding-analysis/data/counties_combined_dfs/combined_vars_with_nearest_percentiles_{county_name}_{FIPS}.csv",
            index=False,
        )

        # Print the updated combined_vars_df
        print("Combined variables data with nearest percentiles of surge and wave:")
        print(combined_vars_df)

        # Calculate the average percentile of all the drivers for each county based on the conditions of 'surge_points' and 'wave_points'
        combined_vars_df = calculate_average_percentile(combined_vars_df)

        # Save the updated combined_vars_df
        combined_vars_df.to_csv(
            f"{base_path}data/combined_vars_with_average_percentile_{county_name}_{FIPS}.csv",
            index=False,
        )

        # Print the updated combined_vars_df
        print("Combined variables data with average percentile of vars:")
        print(combined_vars_df)

        #############################################################################################################
        ############################# GET THE MAXIMUM AVERAGE OF ALL VARIABLES IN TIME WINDOWs ######################
        #############################################################################################################

        # Print that the maximum average percentile for each location is being calculated
        print("Calculating the maximum average percentile for each location...")

        # Limit all latitudes and longitudes to 6 decimal places
        precip_locations_df["latitude"] = precip_locations_df["latitude"].round(4)
        precip_locations_df["longitude"] = precip_locations_df["longitude"].round(4)
        combined_vars_df["latitude_precip"] = combined_vars_df["latitude_precip"].round(
            4
        )
        combined_vars_df["longitude_precip"] = combined_vars_df[
            "longitude_precip"
        ].round(4)

        # Set Time as index for the combined_vars_df
        combined_vars_df.set_index("Time", inplace=True)

        # Convert the index to datetime format
        combined_vars_df.index = pd.to_datetime(combined_vars_df.index)

        # Initializing the final output dataset
        flood_vars_avg_output_df = pd.DataFrame()

        for i in tqdm(range(len(precip_locations_df))):
            lat = precip_locations_df["latitude"].iloc[i]
            lon = precip_locations_df["longitude"].iloc[i]
            loc_id = precip_locations_df["ID"].iloc[i]

            # Get the data for each location
            df_combined_vars = combined_vars_df[
                (combined_vars_df["latitude_precip"] == lat)
                & (combined_vars_df["longitude_precip"] == lon)
            ]

            # Assign location IDs
            df_combined_vars["Location_ID"] = loc_id

            count = 1

            for i in range(len(sheldus_events_dates)):
                start_date = sheldus_events_dates["Hazard_start"].iloc[i]
                end_date = sheldus_events_dates["Hazard_end"].iloc[i]

                # Get the date range of the window
                first_window_date = start_date - timedelta(days=1)
                last_window_date = end_date + timedelta(days=1)
                window_dates = pd.date_range(first_window_date, last_window_date)

                # Get the values within the window
                window_values_combined_df = df_combined_vars.loc[
                    df_combined_vars.index.intersection(window_dates)
                ]

                # Assign event IDs
                window_values_combined_df["Event_ID"] = count

                # Get the max average_percentile values
                max_avg_values_combined_df = window_values_combined_df[
                    window_values_combined_df["average_percentile"]
                    == window_values_combined_df["average_percentile"].max()
                ]

                # Concatenate to the main dataframe
                flood_vars_avg_output_df = pd.concat(
                    [
                        flood_vars_avg_output_df,
                        max_avg_values_combined_df,
                    ],
                    axis=0,
                )

                count += 1

        # Save the dataset
        flood_vars_avg_output_df.to_csv(
            f"{base_path}data/flood_max_avg_vars_sheldus.csv", index=False
        )

        # Reset the index
        flood_vars_avg_output_df.reset_index(inplace=True)

        # Print the flood output DataFrame
        print("Flood average output DataFrame:")
        print(flood_vars_avg_output_df)

        # Print the columns of the flood output DataFrame
        # print(flood_vars_avg_output_df.columns)

        # Info about the flood output DataFrame
        print("Info about the flood average output DataFrame:")
        print(flood_vars_avg_output_df.info())

        ###########################################################################################################

        # Print that the maximum average percentile in SHELDUS windows for each event is being calculated
        print(
            "Calculating the maximum average percentile in SHELDUS windows for each event..."
        )

        # Extract the maximum average_percentile for each Sheldus window
        flood_vars_output_df = pd.DataFrame()

        for i in tqdm(range(len(sheldus_events_dates))):
            sheldus_id = sheldus_events_dates["ID"].iloc[i]

            # Filter `flood_vars_avg_output_df` to retain only entries corresponding to the current event.
            window_values = flood_vars_avg_output_df.loc[
                flood_vars_avg_output_df.Event_ID == sheldus_id
            ]

            # Add a small random number to avoid ties
            np.random.seed(123)
            window_values["average_percentile"] = window_values[
                "average_percentile"
            ].apply(lambda x: x + np.random.uniform(0, 1) / (10**7))

            # Identify the entry with the maximum average_percentile within the window.
            max_values = window_values[
                window_values["average_percentile"]
                == window_values["average_percentile"].max()
            ]

            # Concatenate the identified entry to `flood_vars_output_df`.
            flood_vars_output_df = pd.concat([flood_vars_output_df, max_values], axis=0)

        flood_vars_output_df.to_csv(
            f"{base_path}data/max_avg_percentile_flood_vars.csv", index=False
        )

        print("Flood vars with max avg percentile output:")
        print(flood_vars_output_df)

        # Add `Hazard_start` and `Hazard_end` columns to `flood_vars_output_df` based on Event_ID and sheldus_events_dates
        # flood_vars_output_df = flood_vars_output_df.merge(
        #     sheldus_events_dates, how="left", left_on="Event_ID", right_on="ID"
        # )

        # Merge the `flood_vars_output_df` with the `sheldus_county_data` to get all the event details
        flood_vars_output_df = flood_vars_output_df.merge(
            sheldus_county_data, how="left", left_on="Event_ID", right_on="ID"
        )

        # Define the output file path
        output_file_path = f"{base_path}/flood_sheldus_df_{county_name}_{FIPS}.csv"

        # Save the final DataFrame
        flood_vars_output_df.to_csv(output_file_path, index=False)

        # # Drop the `ID` column
        # flood_vars_output_df.drop(columns="ID", inplace=True)

        # # Save the final output
        # flood_vars_output_df.to_csv(
        #     f"{base_path}data/flood_clim_vars_max_avg_output_final.csv", index=False
        # )

        # Print the final output
        print("Final flood vars with SHELDUS data output:")
        print(flood_vars_output_df)

        ###########################################################################################################

    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Define the main function
def main():
    """
    This is the main function to execute the flood impact analysis for all counties listed in a CSV file.
    It loads the county information, sets up the necessary file paths, and iterates through each county to
    perform the analysis.

    The function follows these general steps:
    1. Load the county information from a CSV file.
    2. Define the base file paths for various datasets.
    3. Iterate through each county, processing them one by one using the `process_county` function.

    Parameters:
    None

    Returns:
    None

    Raises:
    FileNotFoundError: If the CSV file with county information is not found.
    Exception: For any other issues that might occur during the processing.
    """
    try:
        # Load the county information from a CSV file
        county_info_df = pd.read_csv(
            "../data/ready-for-analysis/sheldus/county_names_fips_codes_states_coasts.csv"
        )

        # Define the file paths for various datasets
        SHELDUS_COUNTIES_PATH = "../data/ready-data/sheldus-counties/"
        SHELDUS_SAVE_PATH = "../data/ready-for-analysis/sheldus/"
        STORM_SURGE_PATH = (
            "../data/ready-for-analysis/storm-surge/combined_ts_surge-percentiles.csv"
        )
        WATER_LEVEL_PATH = "../data/ready-for-analysis/storm-surge/combined_ts_waterlevel-percentiles.csv"
        WAVE_PATH = "../data/ready-for-analysis/waves/"
        SAVE_PATH = "../flood-impact-analysis/"

        # Process each county one by one
        for index, row in tqdm(
            county_info_df.iterrows(), total=county_info_df.shape[0]
        ):
            process_county(
                row,
                SHELDUS_COUNTIES_PATH,
                SHELDUS_SAVE_PATH,
                STORM_SURGE_PATH,
                WATER_LEVEL_PATH,
                WAVE_PATH,
                SAVE_PATH,
            )

    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Execute the main function
if __name__ == "__main__":
    main()


# Print the total time taken for the analysis
print(f"Total time taken: {datetime.now() - start_time}")

# Print that the script is complete
print("Script completed successfully!")

# End of script
