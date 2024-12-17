import os
import warnings

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from geopandas.tools import sjoin
from matplotlib.colors import BoundaryNorm, ListedColormap

warnings.filterwarnings("ignore")

# File paths and URLs
DATA_FILE = "../final_df_95th_coastal.csv"
DATA_FILE_median = "../combined_flood_sheldus_data_with_compound_events_coastal.csv"
SHAPEFILE_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2019/COUNTY/tl_2019_us_county.zip"
)
SHAPEFILE_STATE_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2019/STATE/tl_2019_us_state.zip"
)
COAST_INFO_FILE = (
    "../data/county_names_fips_codes_states_coasts_complete_names_updated_final_v2.csv"
)


def load_and_preprocess_data(file_path, file_path_median):
    """
    Load and preprocess the flooding event data.

    Args:
        file_path (str): Path to the CSV file containing flooding event data.
        file_path_median (str): Path to the CSV file containing median event data.

    Returns:
        pd.DataFrame, pd.DataFrame: Preprocessed DataFrame with calculated loss ratios and a new DataFrame with median metrics.
    """
    # Load the data
    df = pd.read_csv(file_path)
    df_median = pd.read_csv(file_path_median)

    # Ensure FIPS codes are strings and appropriately formatted with leading zeros
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df_median["fips_code"] = df_median["fips_code"].astype(str).str.zfill(5)

    # Calculate non-CF losses
    df["non_cf_loss_property_damage"] = (
        df["total_loss_property_damage"] - df["compound_loss_property_damage"]
    )
    df["non_cf_loss_crop_damage"] = (
        df["total_loss_crop_damage"] - df["compound_loss_crop_damage"]
    )
    df["non_cf_loss_injuries"] = (
        df["total_loss_injuries"] - df["compound_loss_injuries"]
    )
    df["non_cf_loss_fatalities"] = (
        df["total_loss_fatalities"] - df["compound_loss_fatalities"]
    )

    # Normalize the losses by the overall flood losses in each county
    for loss_type in ["property_damage", "crop_damage", "injuries", "fatalities"]:
        df[f"norm_compound_loss_{loss_type}"] = (
            df[f"compound_loss_{loss_type}"] / df[f"total_loss_{loss_type}"]
        )
        df[f"norm_non_cf_loss_{loss_type}"] = (
            df[f"non_cf_loss_{loss_type}"] / df[f"total_loss_{loss_type}"]
        )

    # Calculate the number of non-CF events
    df["non_cf_events"] = df["total_events"] - df["number_of_compound_events"]

    # Calculate average losses per event
    for loss_type in ["property_damage", "crop_damage", "injuries", "fatalities"]:
        cf_column = f"compound_loss_{loss_type}"
        non_cf_column = f"non_cf_loss_{loss_type}"

        df[f"avg_cf_loss_{loss_type}"] = df[cf_column] / df["number_of_compound_events"]
        df[f"avg_non_cf_loss_{loss_type}"] = df[non_cf_column] / df["non_cf_events"]

    # CF median calculations
    cf_median_property = (
        df_median[(df_median["Event_Type"] == "CF") & (df_median["PropDmgAdj"] != 0)]
        .groupby(["fips_code", "County"])
        .agg({"PropDmgAdj": "median"})
        .rename(columns={"PropDmgAdj": "median_cf_loss_property_damage"})
        .reset_index()
    )

    cf_median_crop = (
        df_median[(df_median["Event_Type"] == "CF") & (df_median["CropDmgAdj"] != 0)]
        .groupby(["fips_code", "County"])
        .agg({"CropDmgAdj": "median"})
        .rename(columns={"CropDmgAdj": "median_cf_loss_crop_damage"})
        .reset_index()
    )

    cf_median = pd.merge(
        cf_median_property, cf_median_crop, on=["fips_code", "County"], how="outer"
    )

    # Non-CF median calculations
    non_cf_median_property = (
        df_median[(df_median["Event_Type"] != "CF") & (df_median["PropDmgAdj"] != 0)]
        .groupby(["fips_code", "County"])
        .agg({"PropDmgAdj": "median"})
        .rename(columns={"PropDmgAdj": "median_non_cf_loss_property_damage"})
        .reset_index()
    )

    non_cf_median_crop = (
        df_median[(df_median["Event_Type"] != "CF") & (df_median["CropDmgAdj"] != 0)]
        .groupby(["fips_code", "County"])
        .agg({"CropDmgAdj": "median"})
        .rename(columns={"CropDmgAdj": "median_non_cf_loss_crop_damage"})
        .reset_index()
    )

    non_cf_median = pd.merge(
        non_cf_median_property,
        non_cf_median_crop,
        on=["fips_code", "County"],
        how="outer",
    )

    # Create a new DataFrame to store the combined median metrics
    median_metrics_df = pd.merge(
        cf_median, non_cf_median, on=["fips_code", "County"], how="outer"
    )

    # Calculate median loss ratios (CF / non-CF)
    median_metrics_df["median_loss_ratio_property_damage"] = (
        median_metrics_df["median_cf_loss_property_damage"]
        / median_metrics_df["median_non_cf_loss_property_damage"]
    )
    median_metrics_df["median_loss_ratio_crop_damage"] = (
        median_metrics_df["median_cf_loss_crop_damage"]
        / median_metrics_df["median_non_cf_loss_crop_damage"]
    )

    # Calculate loss ratios (CF / non-CF)
    for loss_type in ["property_damage", "crop_damage", "injuries", "fatalities"]:
        df[f"avg_loss_ratio_{loss_type}"] = (
            df[f"avg_cf_loss_{loss_type}"] / df[f"avg_non_cf_loss_{loss_type}"]
        )

    # Replace infinity and NaN values with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Replace infinity and NaN values with 0 in the median metrics DataFrame
    median_metrics_df = median_metrics_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Save the median metrics data to a CSV file
    median_metrics_df.to_csv("../median_metrics_data.csv", index=False)

    return df, median_metrics_df


def prepare_data_for_map(county_data, event_data):
    """
    Prepares the data by merging the event data with the county shapefiles.

    Parameters:
        county_data (GeoDataFrame): The geopandas GeoDataFrame with U.S. county boundaries.
        event_data (DataFrame): The pandas DataFrame with event data.

    Returns:
        GeoDataFrame: A merged geopandas GeoDataFrame ready for plotting.
    """
    # Ensure FIPS codes are strings and appropriately formatted with leading zeros
    event_data["FIPS"] = event_data["FIPS"].astype(str).str.zfill(5)
    county_data["GEOID"] = county_data["GEOID"].astype(str)

    # Merge the DataFrames on the FIPS code
    merged_data = county_data.merge(event_data, left_on="GEOID", right_on="FIPS")

    # Reproject the geometries to the "Web Mercator" coordinate system
    merged_data = merged_data.to_crs(epsg=3857)

    # Calculate the centroids of the counties
    merged_data["centroid"] = merged_data.geometry.centroid

    # Reproject the centroids back to the original coordinate system
    merged_data["centroid"] = merged_data["centroid"].to_crs(epsg=4326)

    return merged_data


def plot_crop_median_loss_ratio_map(
    merged_data,
    shapefile_url,
    shapefile_state_url,
    figure_path,
    number,
    figsize=(22, 18),
):
    """
    Plots a choropleth map of the U.S. counties colored by the median crop loss ratio.

    Parameters:
        merged_data (GeoDataFrame): The merged GeoDataFrame containing event data and geography.
        shapefile_url (str): The URL to the shapefile containing U.S. county boundaries.
        shapefile_state_url (str): The URL to the shapefile containing U.S. state boundaries.
        figure_path (str): Path to save the output map.
        number (int): Identifier for the saved figure.
        figsize (tuple): The figure size (default: (22, 18)).
    """
    column = "median_loss_ratio_crop_damage"
    median_non_cf_column = "median_non_cf_loss_crop_damage"
    median_cf_column = "median_cf_loss_crop_damage"

    # Load the USA shapefile into a GeoDataFrame
    usa = gpd.read_file(shapefile_url)
    usa_state = gpd.read_file(shapefile_state_url)

    # Reproject the geometries to the "Web Mercator" coordinate system
    usa = usa.to_crs(epsg=3857)
    usa_state = usa_state.to_crs(epsg=3857)

    # Calculate the centroids of the counties
    usa["centroid"] = usa.geometry.centroid
    usa_state["centroid"] = usa_state.geometry.centroid

    # Reproject the centroids back to the original coordinate system
    usa["centroid"] = usa["centroid"].to_crs(epsg=4326)
    usa_state["centroid"] = usa_state["centroid"].to_crs(epsg=4326)

    # Filter the counties based on the longitude of the centroids
    eastern_usa = usa[
        (usa["centroid"].x > -100) & (usa["centroid"].y > 24) & (usa["centroid"].y < 50)
    ]
    eastern_usa_state = usa_state[
        (usa_state["centroid"].x > -100)
        & (usa_state["centroid"].y > 24)
        & (usa_state["centroid"].y < 50)
    ]

    # Reproject merged_data to the same CRS as eastern_usa
    merged_data = merged_data.to_crs(eastern_usa.crs)

    # Perform a spatial join between the state and county GeoDataFrames
    states_with_counties = sjoin(
        eastern_usa_state, merged_data, how="inner", predicate="intersects"
    )

    # Plotting the map
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    above_100_df = merged_data[
        (merged_data[column] > 50)
        | (np.isinf(merged_data[column]))
        | (merged_data[median_non_cf_column] == 0)
    ]

    yellow_df = merged_data[
        (merged_data[column] > 0) & (merged_data[column] <= 1)
        | (
            (merged_data[median_cf_column] == 0)
            & (merged_data[median_non_cf_column] != 0)
        )
    ]

    continuous_df = merged_data[(merged_data[column] > 1) & (merged_data[column] <= 50)]

    # Define the color intervals and the corresponding colormap
    boundaries = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cmap = plt.colormaps["rocket_r"]
    last_color = cmap(1.0)
    cmap = ListedColormap(cmap(np.linspace(0, 1, len(boundaries) - 1)))
    norm = BoundaryNorm(boundaries, ncolors=len(boundaries), extend="max")

    # Create a colorbar to indicate overflow
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.0001, extend="max")
    cbar.set_label("Crop Loss Ratio (CF/non-CF)", fontsize=20, rotation=90, labelpad=10)
    cbar.set_ticks(boundaries)
    cbar.ax.set_yticklabels([str(b) for b in boundaries], fontsize=20)

    # Plot the states with county information
    states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

    # Plot yellow counties (0 to 1)
    if not yellow_df.empty:
        yellow_df.plot(ax=ax, color="yellow", linewidth=0.5, edgecolor="0.8")

    # Plot counties with values between 1 and 50 using continuous color map
    if not continuous_df.empty:
        continuous_df.plot(
            column=column,
            ax=ax,
            legend=False,
            cmap=cmap,
            norm=norm,
            linewidth=0.5,
            edgecolor="0.8",
        )

    # Plot counties with values above 50
    if not above_100_df.empty:
        above_100_df.plot(
            ax=ax,
            legend=False,
            color="#03051A",
            linewidth=0.5,
            edgecolor="0.8",
        )

    # Gray out counties where total_loss_crop_damage is 0
    gray_df = merged_data[merged_data["total_loss_crop_damage"] == 0]
    if not gray_df.empty:
        gray_df.plot(ax=ax, color="lightgray", linewidth=0.5, edgecolor="0.8")

    # Add legend for yellow and gray colors
    yellow_patch = mpatches.Patch(color="yellow", label="Ratio between 0 and 1")
    gray_patch = mpatches.Patch(color="lightgray", label="No losses from flooding")
    legend = ax.legend(
        handles=[yellow_patch, gray_patch],
        loc="center left",
        bbox_to_anchor=(0.68, 0.3),
        fontsize=16,
    )

    # Set the title of the plot
    ax.set_title(
        f'{column.replace("_", " ").title()}',
        loc="center",
        fontweight="bold",
        fontsize=24,
    )

    # Remove axis spines
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Calculate the number of counties with ratio above 50 (adjusted) by excluding counties with total_loss_crop_damage = 0
    adjusted_above_50 = above_100_df[(above_100_df["total_loss_crop_damage"] != 0)]

    # Print the number of counties in adjusted_above_50
    print(
        f"\nCHECK: No. of counties with ratio above 50 (adjusted): {adjusted_above_50.shape[0]}"
    )

    # Combine above_100_df and continuous_df dataframes into one
    combined_df = pd.concat([adjusted_above_50, continuous_df])

    # Calculate the average of the crop median loss ratios
    average_loss_ratio = combined_df["median_loss_ratio_crop_damage"].mean()

    # Print the number of counties in each category
    print(f"\nMedian Loss Ratios: {column}")
    adjusted_above_50_count = above_100_df.shape[0] - gray_df.shape[0]
    print(f"No. of counties with ratio above 50 (adjusted): {adjusted_above_50_count}")
    print(f"No. of counties with ratio between 0 and 1: {yellow_df.shape[0]}")
    print(f"No. of counties with ratio between 1 and 50: {continuous_df.shape[0]}")
    print(
        f"Average crop median loss ratio (for ratio > 1): {round(average_loss_ratio, 2)}"
    )

    # Save the plot
    # plt.savefig(f"{figure_path}map_{column}_{number}.png", dpi=500, bbox_inches="tight")
    # plt.savefig(f"{figure_path}map_{column}_{number}.svg", bbox_inches="tight")
    plt.close()


def plot_property_median_loss_ratio_map(
    merged_data,
    shapefile_url,
    shapefile_state_url,
    figure_path,
    number,
    figsize=(22, 18),
):
    """
    Plots a choropleth map of the U.S. counties colored by the median property loss ratio.

    Parameters:
        merged_data (GeoDataFrame): The merged GeoDataFrame containing event data and geography.
        shapefile_url (str): The URL to the shapefile containing U.S. county boundaries.
        shapefile_state_url (str): The URL to the shapefile containing U.S. state boundaries.
        figure_path (str): Path to save the output map.
        number (int): Identifier for the saved figure.
        figsize (tuple): The figure size (default: (22, 18)).
    """
    column = "median_loss_ratio_property_damage"
    median_non_cf_column = "median_non_cf_loss_property_damage"
    median_cf_column = "median_cf_loss_property_damage"

    # Load the USA shapefile into a GeoDataFrame
    usa = gpd.read_file(shapefile_url)
    usa_state = gpd.read_file(shapefile_state_url)

    # Reproject the geometries to the "Web Mercator" coordinate system
    usa = usa.to_crs(epsg=3857)
    usa_state = usa_state.to_crs(epsg=3857)

    # Calculate the centroids of the counties
    usa["centroid"] = usa.geometry.centroid
    usa_state["centroid"] = usa_state.geometry.centroid

    # Reproject the centroids back to the original coordinate system
    usa["centroid"] = usa["centroid"].to_crs(epsg=4326)
    usa_state["centroid"] = usa_state["centroid"].to_crs(epsg=4326)

    # Filter the counties based on the longitude of the centroids
    eastern_usa = usa[
        (usa["centroid"].x > -100) & (usa["centroid"].y > 24) & (usa["centroid"].y < 50)
    ]
    eastern_usa_state = usa_state[
        (usa_state["centroid"].x > -100)
        & (usa_state["centroid"].y > 24)
        & (usa_state["centroid"].y < 50)
    ]

    # Reproject merged_data to the same CRS as eastern_usa
    merged_data = merged_data.to_crs(eastern_usa.crs)

    # Perform a spatial join between the state and county GeoDataFrames
    states_with_counties = sjoin(
        eastern_usa_state, merged_data, how="inner", predicate="intersects"
    )

    # Plotting the map
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    above_100_df = merged_data[
        (merged_data[column] > 50)
        | (np.isinf(merged_data[column]))
        | (merged_data[median_non_cf_column] == 0)
    ]

    yellow_df = merged_data[(merged_data[column] > 0) & (merged_data[column] <= 1)]

    continuous_df = merged_data[(merged_data[column] > 1) & (merged_data[column] <= 50)]

    # Define the color intervals and the corresponding colormap
    boundaries = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cmap = plt.colormaps["rocket_r"]
    last_color = cmap(1.0)
    cmap = ListedColormap(cmap(np.linspace(0, 1, len(boundaries) - 1)))
    norm = BoundaryNorm(boundaries, ncolors=len(boundaries), extend="max")

    # Create a colorbar to indicate overflow
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.0001, extend="max")
    cbar.set_label(
        "Property Loss Ratio (CF/non-CF)", fontsize=20, rotation=90, labelpad=10
    )
    cbar.set_ticks(boundaries)
    cbar.ax.set_yticklabels([str(b) for b in boundaries], fontsize=20)

    # Plot the states with county information
    states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

    # Plot yellow counties (0 to 1)
    if not yellow_df.empty:
        yellow_df.plot(ax=ax, color="yellow", linewidth=0.5, edgecolor="0.8")

    # Plot counties with values between 1 and 50 using continuous color map
    if not continuous_df.empty:
        continuous_df.plot(
            column=column,
            ax=ax,
            legend=False,
            cmap=cmap,
            norm=norm,
            linewidth=0.5,
            edgecolor="0.8",
        )

    # Plot counties with values above 50
    if not above_100_df.empty:
        above_100_df.plot(
            ax=ax,
            legend=False,
            color="#03051A",
            linewidth=0.5,
            edgecolor="0.8",
        )

    # Add legend for yellow color (0 to 1)
    yellow_patch = mpatches.Patch(color="yellow", label="Ratio between 0 and 1")
    legend = ax.legend(
        handles=[yellow_patch],
        loc="center left",
        bbox_to_anchor=(0.68, 0.3),
        fontsize=16,
    )

    # Set the title of the plot
    ax.set_title(
        f'{column.replace("_", " ").title()}',
        loc="center",
        fontweight="bold",
        fontsize=24,
    )

    # Remove axis spines
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Combine above_100_df and continuous_df dataframes into one
    combined_df = pd.concat([above_100_df, continuous_df])

    # Calculate the average of the property median loss ratios
    average_loss_ratio = combined_df["median_loss_ratio_property_damage"].mean()

    # Print the number of counties in each category
    print(f"\nMedian Loss Ratios: {column}")
    print(f"No. of counties with ratio above 50: {above_100_df.shape[0]}")
    print(f"No. of counties with ratio between 0 and 1: {yellow_df.shape[0]}")
    print(f"No. of counties with ratio between 1 and 50: {continuous_df.shape[0]}")
    print(
        f"Average property median loss ratio (for ratio > 1): {round(average_loss_ratio, 2)}"
    )

    # Save the plot
    # plt.savefig(f"{figure_path}map_{column}_{number}.png", dpi=500, bbox_inches="tight")
    # plt.savefig(f"{figure_path}map_{column}_{number}.svg", bbox_inches="tight")
    plt.close()


def plot_average_loss_ratio_map(
    merged_data,
    shapefile_url,
    shapefile_state_url,
    figure_path,
    number,
    figsize=(22, 18),
):
    """
    Plots a choropleth map of the U.S. counties colored by the average loss ratio.

    Parameters:
        merged_data (GeoDataFrame): The merged GeoDataFrame containing event data and geography.
        shapefile_url (str): The URL to the shapefile containing U.S. county boundaries.
        shapefile_state_url (str): The URL to the shapefile containing U.S. state boundaries.
        figure_path (str): Path to save the output map.
        number (int): Identifier for the saved figure.
        figsize (tuple): The figure size (default: (22, 18)).
    """
    column = "avg_loss_ratio_property_damage"
    avg_non_cf_column = "avg_non_cf_loss_property_damage"

    # Load the USA shapefile into a GeoDataFrame
    usa = gpd.read_file(shapefile_url)
    usa_state = gpd.read_file(shapefile_state_url)

    # Reproject the geometries to the "Web Mercator" coordinate system
    usa = usa.to_crs(epsg=3857)
    usa_state = usa_state.to_crs(epsg=3857)

    # Calculate the centroids of the counties
    usa["centroid"] = usa.geometry.centroid
    usa_state["centroid"] = usa_state.geometry.centroid

    # Reproject the centroids back to the
    usa["centroid"] = usa["centroid"].to_crs(epsg=4326)
    usa_state["centroid"] = usa_state["centroid"].to_crs(epsg=4326)

    # Filter the counties based on the longitude of the centroids
    eastern_usa = usa[
        (usa["centroid"].x > -100) & (usa["centroid"].y > 24) & (usa["centroid"].y < 50)
    ]
    eastern_usa_state = usa_state[
        (usa_state["centroid"].x > -100)
        & (usa_state["centroid"].y > 24)
        & (usa_state["centroid"].y < 50)
    ]

    # Reproject merged_data to the same CRS as eastern_usa
    merged_data = merged_data.to_crs(eastern_usa.crs)

    # Perform a spatial join between the state and county GeoDataFrames
    states_with_counties = sjoin(
        eastern_usa_state, merged_data, how="inner", predicate="intersects"
    )

    # Plotting the map
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    above_100_df = merged_data[
        (merged_data[column] > 100)
        | (np.isinf(merged_data[column]))
        | (merged_data[avg_non_cf_column] == 0)
    ]

    yellow_df = merged_data[(merged_data[column] > 0) & (merged_data[column] <= 1)]

    continuous_df = merged_data[
        (merged_data[column] > 1) & (merged_data[column] <= 100)
    ]

    # Define the color intervals and the corresponding colormap
    boundaries = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cmap = plt.colormaps["rocket_r"]
    last_color = cmap(1.0)
    cmap = ListedColormap(cmap(np.linspace(0, 1, len(boundaries) - 1)))
    norm = BoundaryNorm(boundaries, ncolors=len(boundaries), extend="max")

    # Create a colorbar to indicate overflow
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.0001, extend="max")
    cbar.set_label(
        "Property Loss Ratio (CF/non-CF)", fontsize=20, rotation=90, labelpad=10
    )
    cbar.ax.set_yticklabels(
        ["1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"],
        fontsize=20,
    )

    # Plot the states with county information
    states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

    # Plot yellow counties (0 to 1)
    if not yellow_df.empty:
        yellow_df.plot(ax=ax, color="yellow", linewidth=0.5, edgecolor="0.8")

    # Plot counties with values between 1 and 100 using continuous color map
    if not continuous_df.empty:
        continuous_df.plot(
            column=column,
            ax=ax,
            legend=False,
            cmap=cmap,
            norm=norm,
            linewidth=0.5,
            edgecolor="0.8",
        )

    # Plot counties with values above 100
    if not above_100_df.empty:
        above_100_df.plot(
            ax=ax,
            legend=False,
            color="#03051A",
            linewidth=0.5,
            edgecolor="0.8",
        )

    # Add legend for yellow color (0 to 1)
    yellow_patch = mpatches.Patch(color="yellow", label="Ratio between 0 and 1")
    legend = ax.legend(
        handles=[yellow_patch],
        loc="center left",
        bbox_to_anchor=(0.75, 0.3),
        fontsize=16,
    )

    # Set the title of the plot
    ax.set_title(
        f'{column.replace("_", " ").title()}',
        loc="center",
        fontweight="bold",
        fontsize=24,
    )

    # Remove axis spines
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Print the number of counties in each category
    print(f"\nAverage Loss Ratios: {column}")
    print(f"No. of counties with ratio above 100: {above_100_df.shape}")
    print(f"No. of counties with ratio between 0 and 1: {yellow_df.shape}")
    print(f"No. of counties with ratio between 1 and 100: {continuous_df.shape}")

    # Save the plot
    # plt.savefig(f"{figure_path}map_{column}_{number}.png", dpi=500, bbox_inches="tight")
    # plt.savefig(f"{figure_path}map_{column}_{number}.svg", bbox_inches="tight")
    plt.close()


def plot_loss_ratio_map(
    merged_data,
    column,
    shapefile_url,
    shapefile_state_url,
    figure_path,
    number,
    figsize=(22, 18),
):
    """
    Plots a choropleth map of the U.S. counties colored by the specified column with specific color criteria.

    Parameters:
        merged_data (GeoDataFrame): The merged GeoDataFrame containing event data and geography.
        column (str): The column name to use for coloring the counties.
        shapefile_url (str): The URL to the shapefile containing U.S. county boundaries.
        shapefile_state_url (str): The URL to the shapefile containing U.S. state boundaries.
        figure_path (str): Path to save the output map.
        figsize (tuple): The figure size (default: (22, 18)).
    """

    # Load the USA shapefile into a GeoDataFrame
    usa = gpd.read_file(shapefile_url)
    usa_state = gpd.read_file(shapefile_state_url)

    # Reproject the geometries to the "Web Mercator" coordinate system
    usa = usa.to_crs(epsg=3857)
    usa_state = usa_state.to_crs(epsg=3857)

    # Calculate the centroids of the counties
    usa["centroid"] = usa.geometry.centroid
    usa_state["centroid"] = usa_state.geometry.centroid

    # Reproject the centroids back to the original coordinate system
    usa["centroid"] = usa["centroid"].to_crs(epsg=4326)
    usa_state["centroid"] = usa_state["centroid"].to_crs(epsg=4326)

    # Filter the counties based on the longitude of the centroids
    eastern_usa = usa[
        (usa["centroid"].x > -100) & (usa["centroid"].y > 24) & (usa["centroid"].y < 50)
    ]
    eastern_usa_state = usa_state[
        (usa_state["centroid"].x > -100)
        & (usa_state["centroid"].y > 24)
        & (usa_state["centroid"].y < 50)
    ]

    # Reproject merged_data to the same CRS as eastern_usa
    merged_data = merged_data.to_crs(eastern_usa.crs)

    # Perform a spatial join between the state and county GeoDataFrames
    states_with_counties = sjoin(
        eastern_usa_state, merged_data, how="inner", predicate="intersects"
    )

    # Plotting the map
    # fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Determine the layout based on the column name
    if column.startswith("median"):
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        median_columns = [
            "median_loss_ratio_property_damage",
            "median_loss_ratio_crop_damage",
        ]
        median_non_cf_columns = [
            "median_non_cf_loss_property_damage",
            "median_non_cf_loss_crop_damage",
        ]
        median_cf_columns = [
            "median_cf_loss_property_damage",
            "median_cf_loss_crop_damage",
        ]

        for ax, median_column, median_non_cf_column, median_cf_column in zip(
            axes, median_columns, median_non_cf_columns, median_cf_columns
        ):
            above_100_df = merged_data[
                (
                    (
                        (merged_data[median_column] > 50)
                        | (np.isinf(merged_data[median_column]))
                    )
                    | (merged_data[median_non_cf_column] == 0)
                )
            ]  # Counties with ratio above 100 for median ratio or ratio 0 but non-CF median loss property damage not 0

            if median_column == "median_loss_ratio_crop_damage":
                yellow_df = merged_data[
                    (
                        (merged_data[median_column] > 0)
                        & (merged_data[median_column] <= 1)
                    )
                    | (
                        (merged_data[median_cf_column] == 0)
                        & (merged_data[median_non_cf_column] != 0)
                    )
                ]  # Counties with ratio between 0 and 1 or ratio 0 but non-CF average loss crop damage not 0

            else:
                yellow_df = merged_data[
                    (
                        (merged_data[median_column] > 0)
                        & (merged_data[median_column] <= 1)
                    )
                ]  # Counties with ratio between 0 and 1 or ratio 0 but non-CF average loss property damage not 0

            # yellow_df = merged_data[
            #     ((merged_data[column] > 0) & (merged_data[column] <= 1))
            # ]  # Counties with ratio between 0 and 1 or ratio 0 but non-CF average loss property damage not
            continuous_df = merged_data[
                (merged_data[median_column] > 1) & (merged_data[median_column] <= 50)
            ]  # Counties with ratio between 1 and 100

            # Define the color intervals and the corresponding colormap
            boundaries = [
                1,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
            ]  # Add intervals with a step of 5
            cmap = plt.colormaps["rocket_r"]  # Access the colormap using the new syntax
            last_color = cmap(1.0)  # Get the last color in the colormap
            cmap = ListedColormap(
                cmap(np.linspace(0, 1, len(boundaries) - 1))
            )  # Create a ListedColormap with the correct number of colors
            norm = BoundaryNorm(boundaries, ncolors=len(boundaries), extend="max")

            # Create a colorbar to indicate overflow
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.0001, extend="max")
            # Set the colorbar title based on the median_column
            if "property" in median_column:
                cbar.set_label(
                    "Property Loss Ratio (CF/non-CF)",
                    fontsize=20,
                    rotation=90,
                    labelpad=10,
                )
            elif "crop" in median_column:
                cbar.set_label(
                    "Crop Loss Ratio (CF/non-CF)",
                    fontsize=20,
                    rotation=90,
                    labelpad=10,
                )

            cbar.set_ticks(boundaries)  # Set the ticks to match the boundaries
            cbar.ax.set_yticklabels(
                [str(b) for b in boundaries],
                fontsize=20,
            )  # Custom ticks

            # Plot the states with county information
            states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

            # Plot yellow counties (0 to 1)
            if not yellow_df.empty:
                yellow_df.plot(ax=ax, color="yellow", linewidth=0.5, edgecolor="0.8")

            # Plot counties with values between 1 and 100 using continuous color map
            if not continuous_df.empty:
                continuous_df.plot(
                    column=column,
                    ax=ax,
                    legend=False,
                    cmap=cmap,
                    norm=norm,
                    linewidth=0.5,
                    edgecolor="0.8",
                )

            # Plot counties with values above 100
            if not above_100_df.empty:
                above_100_df.plot(
                    # column=column,
                    ax=ax,
                    legend=False,
                    color="#03051A",  # last_color,
                    # cmap=cmap,
                    # norm=norm,
                    linewidth=0.5,
                    edgecolor="0.8",
                )

            # Add legend for yellow color (0 to 1)
            # yellow_patch = mpatches.Patch(color="yellow", label="Ratio between 0 and 1")
            # legend = ax.legend(
            #     handles=[yellow_patch],
            #     loc="center left",
            #     bbox_to_anchor=(0.65, 0.3),
            #     fontsize=16,
            # )

            # Gray out counties where total_loss_crop_damage is 0 for crop damage subplot
            if median_column == "median_loss_ratio_crop_damage":
                gray_df = merged_data[merged_data["total_loss_crop_damage"] == 0]
                if not gray_df.empty:
                    gray_df.plot(
                        ax=ax, color="lightgray", linewidth=0.5, edgecolor="0.8"
                    )

                # Add legend for gray color (no losses from flooding)
                gray_patch = mpatches.Patch(
                    color="lightgray", label="No losses from flooding"
                )
                yellow_patch = mpatches.Patch(
                    color="yellow", label="Ratio between 0 and 1"
                )
                legend = ax.legend(
                    handles=[yellow_patch, gray_patch],
                    loc="center left",
                    bbox_to_anchor=(0.68, 0.3),
                    fontsize=16,
                )
            else:
                # Add legend for yellow color (0 to 1)
                yellow_patch = mpatches.Patch(
                    color="yellow", label="Ratio between 0 and 1"
                )
                legend = ax.legend(
                    handles=[yellow_patch],
                    loc="center left",
                    bbox_to_anchor=(0.68, 0.3),
                    fontsize=16,
                )

            # Set the title of the plot for both subplots
            ax.set_title(
                f'{median_column.replace("_", " ").title()}',
                loc="center",
                fontweight="bold",
                fontsize=24,
            )

            # Remove axis spines
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # Hide axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

            plt.tight_layout()

    # Filter the data based on the column
    # if column == "median_loss_ratio_property_damage":
    #     # Define the column names for non-CF loss property damage
    #     median_non_cf_column = "median_non_cf_loss_property_damage"

    #     above_100_df = merged_data[
    #         (
    #             ((merged_data[column] > 50) | (np.isinf(merged_data[column])))
    #             | (merged_data[median_non_cf_column] == 0)
    #         )
    #     ]  # Counties with ratio above 100 for median ratio or ratio 0 but non-CF median loss property damage not 0

    #     yellow_df = merged_data[
    #         ((merged_data[column] > 0) & (merged_data[column] <= 1))
    #         # | ((merged_data[column] == 0) & (merged_data[avg_non_cf_column] != 0))
    #     ]  # Counties with ratio between 0 and 1 or ratio 0 but non-CF average loss property damage not
    #     continuous_df = merged_data[
    #         (merged_data[column] > 1) & (merged_data[column] <= 50)
    #     ]  # Counties with ratio between 1 and 100

    #     # Define the color intervals and the corresponding colormap
    #     boundaries = [
    #         1,
    #         5,
    #         10,
    #         15,
    #         20,
    #         25,
    #         30,
    #         35,
    #         40,
    #         45,
    #         50,
    #     ]  # Add intervals with a step of 5
    #     cmap = plt.colormaps["rocket_r"]  # Access the colormap using the new syntax
    #     last_color = cmap(1.0)  # Get the last color in the colormap
    #     cmap = ListedColormap(
    #         cmap(np.linspace(0, 1, len(boundaries) - 1))
    #     )  # Create a ListedColormap with the correct number of colors
    #     norm = BoundaryNorm(boundaries, ncolors=len(boundaries), extend="max")

    #     # Create a colorbar to indicate overflow
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #     sm.set_array([])
    #     cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.0001, extend="max")
    #     cbar.set_label(
    #         "Property Loss Ratio (CF/non-CF)", fontsize=20, rotation=90, labelpad=10
    #     )
    #     cbar.set_ticks(boundaries)  # Set the ticks to match the boundaries
    #     cbar.ax.set_yticklabels(
    #         [str(b) for b in boundaries],
    #         fontsize=20,
    #     )  # Custom ticks
    #     # cbar.ax.set_yticklabels(
    #     #     ["1", "10", "20", "30", "40", "50"],
    #     #     fontsize=20,
    #     # )  # Custom ticks

    #     # Print the number of counties in each category
    #     print(f"\nMedian Loss Ratios: {column}")
    #     print(f"No. of counties with ratio above 50: {above_100_df.shape[0]}")
    #     print(f"No. of counties with ratio between 0 and 1: {yellow_df.shape[0]}")
    #     print(f"No. of counties with ratio between 1 and 50: {continuous_df.shape[0]}")

    elif column == "avg_loss_ratio_property_damage":

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Define the column names for non-CF loss property damage
        avg_non_cf_column = "avg_non_cf_loss_property_damage"
        above_100_df = merged_data[
            (
                ((merged_data[column] > 100) | (np.isinf(merged_data[column])))
                | (merged_data[avg_non_cf_column] == 0)
            )
        ]  # Counties with ratio above 100 for average ratio or ratio 0 but non-CF average loss property damage not 0
        yellow_df = merged_data[
            ((merged_data[column] > 0) & (merged_data[column] <= 1))
            # | ((merged_data[column] == 0) & (merged_data[avg_non_cf_column] != 0))
        ]  # Counties with ratio between 0 and 1 or ratio 0 but non-CF average loss property damage not
        continuous_df = merged_data[
            (merged_data[column] > 1) & (merged_data[column] <= 100)
        ]  # Counties with ratio between 1 and 100

        # Define the color intervals and the corresponding colormap
        boundaries = [
            1,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            # 110,
        ]  # Add an interval for values above 100
        cmap = plt.colormaps["rocket_r"]  # Access the colormap using the new syntax
        last_color = cmap(1.0)  # Get the last color in the colormap
        cmap = ListedColormap(
            cmap(np.linspace(0, 1, len(boundaries) - 1))
        )  # Create a ListedColormap with the correct number of colors
        norm = BoundaryNorm(boundaries, ncolors=len(boundaries), extend="max")

        # Create a colorbar to indicate overflow
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.0001, extend="max")
        cbar.set_label(
            "Property Loss Ratio (CF/non-CF)", fontsize=20, rotation=90, labelpad=10
        )
        # cbar.set_ticks(boundaries)  # Set the ticks to match the boundaries
        cbar.ax.set_yticklabels(
            [
                "1",
                "10",
                "20",
                "30",
                "40",
                "50",
                "60",
                "70",
                "80",
                "90",
                "100",
            ],  # ">100"],
            fontsize=20,
        )  # Custom ticks

        # Print the number of counties in each category
        print(f"\nAverage Loss Ratios: {column}")
        print(f"No. of counties with ratio above 100: {above_100_df.shape[0]}")
        print(f"No. of counties with ratio between 0 and 1: {yellow_df.shape[0]}")
        print(f"No. of counties with ratio between 1 and 100: {continuous_df.shape[0]}")

        # Print data for each category
        # print("\nData for counties with ratio above 100:")
        # print(above_100_df[column])

        # Plot the states with county information
        states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

        # Plot yellow counties (0 to 1)
        if not yellow_df.empty:
            yellow_df.plot(ax=ax, color="yellow", linewidth=0.5, edgecolor="0.8")

        # Plot counties with values between 1 and 100 using continuous color map
        if not continuous_df.empty:
            continuous_df.plot(
                column=column,
                ax=ax,
                legend=False,
                cmap=cmap,
                norm=norm,
                linewidth=0.5,
                edgecolor="0.8",
            )

        # Plot counties with values above 100
        if not above_100_df.empty:
            above_100_df.plot(
                # column=column,
                ax=ax,
                legend=False,
                color="#03051A",  # last_color,
                # cmap=cmap,
                # norm=norm,
                linewidth=0.5,
                edgecolor="0.8",
            )

        # Add legend for yellow color (0 to 1)
        yellow_patch = mpatches.Patch(color="yellow", label="Ratio between 0 and 1")
        legend = plt.legend(
            handles=[yellow_patch],
            loc="center left",
            bbox_to_anchor=(0.75, 0.3),
            fontsize=16,
        )

        # Set the title of the plot
        plt.title(
            f'Map of {column.replace("_", " ").title()}',
            loc="center",
            fontweight="bold",
            fontsize=24,
        )

        # Remove axis spines
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Hide axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

    # Save the plot
    # plt.savefig(
    #     f"{figure_path}map_{column}_{number}_subplot.png",
    #     dpi=500,
    #     bbox_inches="tight",
    # )

    # # Save the plot as an SVG file
    # plt.savefig(
    #     f"{figure_path}map_{column}_{number}_subplot.svg",
    #     bbox_inches="tight",
    # )

    plt.close()


# Load the preprocessed data
# df = load_and_preprocess_data(DATA_FILE, DATA_FILE_median)
df, median_metrics_df = load_and_preprocess_data(DATA_FILE, DATA_FILE_median)

coast_info = pd.read_csv(COAST_INFO_FILE)
coast_info["FIPS"] = coast_info["FIPS"].astype(str).str.zfill(5)
df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
median_metrics_df["fips_code"] = median_metrics_df["fips_code"].astype(str).str.zfill(5)

# Merge the dataframes
merged_df = pd.merge(df, coast_info, on="FIPS", how="left")

# Merge the dataframes with the median metrics
merged_df = pd.merge(
    merged_df, median_metrics_df, left_on="FIPS", right_on="fips_code", how="left"
)

# Check if the columns exist before dropping them
columns_to_drop = ["fips_code_y", "County_y"]
existing_columns_to_drop = [col for col in columns_to_drop if col in merged_df.columns]
merged_df.drop(columns=existing_columns_to_drop, inplace=True)

# Rename columns if they exist
columns_to_rename = {"fips_code_x": "fips_code", "County_x": "County"}
existing_columns_to_rename = {
    k: v for k, v in columns_to_rename.items() if k in merged_df.columns
}
merged_df.rename(columns=existing_columns_to_rename, inplace=True)


# Save the merged data to a CSV file for reference
merged_df.to_csv("../final_df_with_coasts_avg_ratio_95th.csv", index=False)

############################################################################################
#################################### HISTOGRAMS ############################################
############################################################################################
# Define the loss types
loss_types = ["property_damage", "crop_damage"]

# Plot histograms for each loss type and save the figures
for loss_type in loss_types:
    column_name = f"avg_loss_ratio_{loss_type}"

    # Change the style of the plot
    # plt.style.use("science")

    # Define custom bin edges for the histogram to show more detail
    bin_edges = np.arange(0, 500, 5)

    plt.figure(figsize=(10, 6))
    plt.hist(
        df[column_name],
        bins=bin_edges,
        # Edgecolor of the bars
        edgecolor="black",
    )
    plt.title(f'Average Loss Ratio for {loss_type.replace("_", " ").title()}')
    plt.xlabel("Average Loss Ratio ($CF / non-CF$)")
    plt.ylabel("Frequency")
    # plt.grid(False)

    # Make the grid lines lighter
    plt.gca().yaxis.grid(alpha=0.5, linestyle="--", linewidth=0.25, color="gray")
    plt.gca().xaxis.grid(alpha=0.5, linestyle="--", linewidth=0.25, color="gray")

    plt.savefig(
        f"../figures2/avg-contributions/histogram_avg_loss_ratio_{loss_type}.png",
        dpi=400,
        bbox_inches="tight",
    )
    # plt.savefig(f"histogram_avg_loss_ratio_{loss_type}.svg")
    plt.close()

############################################################################################
############################################################################################

# Calculate median loss ratios without considering zero values for the entire region and print the results
median_ratios = {}
for loss_type in loss_types:
    column_name = f"median_loss_ratio_{loss_type}"
    median_ratios[loss_type] = merged_df[merged_df[column_name] > 0][column_name].mean()
    # print(f"Median Loss Ratio ({loss_type}): {median_ratios[loss_type].round(2)}")
    print(
        f"Average of Median Loss Ratio (above 0) ({loss_type}): {round(median_ratios[loss_type], 2)}"
    )

# Calculate the median of median loss ratios for the entire region above 1 and print the results
median_ratios_above_1 = {}
for loss_type in loss_types:
    column_name = f"median_loss_ratio_{loss_type}"
    median_ratios_above_1[loss_type] = merged_df[merged_df[column_name] > 1][
        column_name
    ].mean()
    # print(f"Median Loss Ratio ({loss_type}): {median_ratios[loss_type].round(2)}")
    print(
        f"Average of Median Loss Ratio (above 1) ({loss_type}): {round(median_ratios_above_1[loss_type], 2)}"
    )

# Calculate median of median loss ratios for the entire region
median_median_ratios = {}
for loss_type in loss_types:
    column_name = f"median_loss_ratio_{loss_type}"
    median_median_ratios[loss_type] = merged_df[merged_df[column_name] > 0][
        column_name
    ].median()
    print(
        f"Median of Median Loss Ratio ({loss_type}): {round(median_median_ratios[loss_type], 2)}"
    )


# Calculate the average loss ratios for the entire region
overall_ratios = {}
for loss_type in loss_types:
    column_name = f"avg_loss_ratio_{loss_type}"
    overall_ratios[loss_type] = merged_df[merged_df[column_name] > 0][
        column_name
    ].mean()
    # print(f"Average Loss Ratio ({loss_type}): {overall_ratios[loss_type].round(2)}")
    print(f"Average Loss Ratio ({loss_type}): {round(overall_ratios[loss_type], 2)}")

# Calculate the median average loss ratios for the entire region
median_avg_ratios = {}
for loss_type in loss_types:
    column_name = f"avg_loss_ratio_{loss_type}"
    median_avg_ratios[loss_type] = merged_df[merged_df[column_name] > 0][
        column_name
    ].median()
    print(
        f"Median Average Loss Ratio (above 0) ({loss_type}): {round(median_avg_ratios[loss_type], 2)}"
    )

# Calculate the median average loss ratios for the entire region above 1 and print the results
median_avg_ratios_above_1 = {}
for loss_type in loss_types:
    column_name = f"avg_loss_ratio_{loss_type}"
    median_avg_ratios_above_1[loss_type] = merged_df[merged_df[column_name] > 1][
        column_name
    ].median()
    print(
        f"Median Average Loss Ratio (above 1) ({loss_type}): {round(median_avg_ratios_above_1[loss_type], 2)}"
    )


# Plot the average loss ratios on a map for the entire region
# Define the column to use for coloring the counties
column = "avg_loss_ratio_property_damage"
column_median = "median_loss_ratio_property_damage"

# Define the title of the map
title = "Average Loss Ratio (Property Damage)"
title_median = "Median Loss Ratio (Property Damage)"

# Define the path to save the figure
figure_path = "../figures2/avg-contributions/"

# Check if the directory exists and create it if it doesn't
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

# Load the county shapefile
county_shapefile = gpd.read_file(SHAPEFILE_URL)

# Prepare the data for mapping
prepared_data = prepare_data_for_map(county_shapefile, merged_df)

# Save the prepared data to a CSV file for reference
# prepared_data.to_csv("../prepared_data_avg_median_ratio_95th.csv", index=False)

# Plot the map with the specified criteria
plot_loss_ratio_map(
    prepared_data,
    "avg_loss_ratio_property_damage",
    SHAPEFILE_URL,
    SHAPEFILE_STATE_URL,
    number=100,
    figure_path=figure_path,
)

# Plot the map with the median loss ratios
plot_loss_ratio_map(
    prepared_data,
    "median_loss_ratio_property_damage",
    SHAPEFILE_URL,
    SHAPEFILE_STATE_URL,
    number=50,
    figure_path=figure_path,
)

# Plot the map for median loss ratios of property damage
plot_property_median_loss_ratio_map(
    prepared_data,
    SHAPEFILE_URL,
    SHAPEFILE_STATE_URL,
    figure_path,
    number=50,
)

# Plot the map for median loss ratios of crop damage
plot_crop_median_loss_ratio_map(
    prepared_data,
    SHAPEFILE_URL,
    SHAPEFILE_STATE_URL,
    figure_path,
    number=50,
)

# Print the success message
print("Script completed successfully.")

# END OF THE SCRIPT
