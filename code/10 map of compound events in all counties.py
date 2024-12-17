"""
Compound Flood Events Mapping

Author: Javed Ali
Email: javed.ali@ucf.edu
Date: December 1, 2023

This script creates a geographic map of the US Gulf of Mexico and Atlantic coast counties,
showing the number of compound flood events for each county. The data is filtered to include
only events where the threshold is 0.95.

The script uses geospatial data (shapefile) for US counties and merges it with the flood events data
based on FIPS codes. The counties are then colored based on the number of compound flood events.

Requirements:
- pandas
- geopandas
- matplotlib
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Import libraries
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_data(csv_path):
    """
    Load the CSV data containing flood events information.

    Parameters:
    - csv_path (str): Path to the CSV file

    Returns:
    - DataFrame: Pandas DataFrame containing the loaded data
    """
    return pd.read_csv(csv_path)


def filter_and_aggregate_data(df, threshold=0.95):
    """
    Filter the DataFrame for a specific threshold and aggregate data by FIPS code.

    Parameters:
    - df (DataFrame): The original dataframe with flood events data
    - threshold (float): The threshold value to filter the data

    Returns:
    - DataFrame: Aggregated DataFrame with maximum number of compound events per county
    """
    # Filter for the specified threshold
    threshold_data = df[df["threshold"] == threshold]

    # Aggregate data by FIPS code
    return threshold_data.groupby("FIPS")["percentage_CE"].max().reset_index()


def load_county_shapefile(shapefile_path):
    """
    Load the shapefile for US counties.

    Parameters:
    - shapefile_path (str): Path to the shapefile

    Returns:
    - GeoDataFrame: Geopandas GeoDataFrame containing the shapefile data
    """
    return gpd.read_file(shapefile_path)


def filter_data_for_coastal_counties(df, coastal_states_fips):
    """
    Filter the DataFrame for counties located in coastal states based on their FIPS codes.

    Parameters:
    - df (DataFrame): The original dataframe with flood events data
    - coastal_states_fips (list): List of state FIPS codes along the Gulf of Mexico and Atlantic coasts

    Returns:
    - DataFrame: Filtered DataFrame with data for coastal counties only
    """
    return df[df["STATEFP"].isin(coastal_states_fips)]


def merge_data_with_shapefile(shapefile_df, flood_data_df):
    """
    Merge the shapefile data with flood events data based on FIPS codes.

    Parameters:
    - shapefile_df (GeoDataFrame): GeoDataFrame of the shapefile
    - flood_data_df (DataFrame): DataFrame of the aggregated flood events data

    Returns:
    - GeoDataFrame: Merged GeoDataFrame ready for plotting
    """
    # Convert FIPS codes to integers for matching
    shapefile_df["GEOID"] = shapefile_df["GEOID"].astype(int)
    flood_data_df["FIPS"] = flood_data_df["FIPS"].astype(int)

    # Merge the dataframes
    return shapefile_df.merge(
        flood_data_df, left_on="GEOID", right_on="FIPS", how="inner"
    )


def plot_map(merged_df):
    """
    Plot the map with counties colored by the number of compound flood events.

    Parameters:
    - merged_df (GeoDataFrame): Merged GeoDataFrame containing county shapes and flood data
    """
    # Plotting the map
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    merged_df.plot(
        column="percentage_CE",
        ax=ax,
        legend=True,
        cax=cax,
        linewidth=0.1,
        legend_kwds={"label": "% of Compound Events"},
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            # "hatch": "///",
            "label": "Missing values",
        },
        cmap="YlOrRd",
        edgecolor="black",
    )
    ax.set_title(
        "% of Compound Flood Events (Threshold 0.95) in US Gulf of Mexico and Atlantic Coast Counties",
        fontsize=18,
        fontweight="bold",
        y=1.03,
    )

    # Remove axes
    ax.set_axis_off()

    # Save the figure
    plt.savefig(
        f"{SAVE_FIGURE_PATH}map-compound-events-counties_perc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# List of state FIPS codes along the GoM and Atlantic coasts (from Texas to Maine)
coastal_states_fips = [
    "12",  # Florida
    "13",  # Georgia
    "37",  # North Carolina
    "45",  # South Carolina
    "51",  # Virginia
    "10",  # Delaware
    "24",  # Maryland
    "34",  # New Jersey
    "36",  # New York
    "42",  # Pennsylvania
    "09",  # Connecticut
    "25",  # Massachusetts
    "44",  # Rhode Island
    "33",  # New Hampshire
    "23",  # Maine
    "22",  # Louisiana
    "48",  # Texas
    "01",  # Alabama
    "28",  # Mississippi
]


# Main execution
if __name__ == "__main__":
    # Define paths to the data
    csv_path = "../combined_flooding_events_analysis_all_counties_final.csv"
    shapefile_path = "../data/tl_2023_us_county/tl_2023_us_county.shp"
    SAVE_FIGURE_PATH = "../figures2/"

    # Load and process the data
    flood_data = load_data(csv_path)
    # coastal_flood_data = filter_data_for_coastal_counties(flood_data, coastal_states_fips)
    aggregated_data = filter_and_aggregate_data(flood_data)
    county_shapefile = load_county_shapefile(shapefile_path)
    map_data = merge_data_with_shapefile(county_shapefile, aggregated_data)

    # Filter the counties based on state FIPS codes
    # merged_gdf_coastal = map_data[map_data['STATEFP'].isin(coastal_states_fips)]

    # Plot the map
    plot_map(map_data)


# Print a message to indicate the script has finished
print("Script finished successfully.")


# End of script
