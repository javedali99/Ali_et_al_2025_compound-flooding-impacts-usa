"""
This script visualizes the geographical distribution of clusters for U.S. counties based on clustering results. 
It merges clustering data with U.S. county boundary shapefiles to create a choropleth map, where counties are 
colored according to their cluster assignment. The script is divided into several functions to handle different 
aspects of the process, including data loading, data preparation, and visualization.

Author: Javed Ali
Email: javedali28@gmail.com
Date: May 7, 2024

"""

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from geopandas.tools import sjoin
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D


def load_clustering_data(filepath):
    """
    Loads clustering data from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file containing the clustering results.

    Returns:
        DataFrame: A pandas DataFrame containing the clustering results.
    """
    return pd.read_csv(filepath)


def load_county_shapefiles(url):
    """
    Loads U.S. county shapefiles from a provided URL.

    Parameters:
        url (str): The URL to the shapefile zip archive.

    Returns:
        GeoDataFrame: A geopandas GeoDataFrame containing the U.S. county boundaries.
    """
    return gpd.read_file(url)


def prepare_data(county_data, cluster_data):
    """
    Prepares the data by merging the clustering data with the county shapefiles.

    Parameters:
        county_data (GeoDataFrame): The geopandas GeoDataFrame with U.S. county boundaries.
        cluster_data (DataFrame): The pandas DataFrame with cluster assignments.

    Returns:
        GeoDataFrame: A merged geopandas GeoDataFrame ready for plotting.
    """
    # Ensure FIPS codes are integers and appropriately formatted with leading zeros
    cluster_data["FIPS"] = (
        cluster_data["FIPS"].astype(int).astype(str).str.zfill(5).str.strip()
    )
    county_data["GEOID"] = county_data["GEOID"].astype(str).str.strip()

    print(cluster_data["FIPS"].unique())
    print(county_data["GEOID"].unique())

    # Merge the DataFrames on the FIPS code
    merged_data = county_data.merge(cluster_data, left_on="GEOID", right_on="FIPS")

    common_values = set(cluster_data["FIPS"]).intersection(set(county_data["GEOID"]))
    print(f"Number of common values: {len(common_values)}")

    # Check if the merged data is empty
    if merged_data.empty:
        raise ValueError("Merged data is empty. Check your input data.")

    # Reproject the geometries to the "Web Mercator" coordinate system
    merged_data = merged_data.to_crs(epsg=3857)

    # Calculate the centroids of the counties
    merged_data["centroid"] = merged_data.geometry.centroid

    # Reproject the centroids back to the original coordinate system
    merged_data["centroid"] = merged_data["centroid"].to_crs(epsg=4326)

    # Filter the counties based on the longitude of the centroids
    eastern_usa = merged_data[merged_data["centroid"].x > -100]

    return eastern_usa


def plot_clusters(merged_data, shapefile_url, shapefile_state_url):
    """
    Plots the U.S. counties colored by their cluster assignments.

    Parameters:
        merged_data (GeoDataFrame): The merged GeoDataFrame containing cluster assignments and geography.

    Side Effects:
        Displays a matplotlib plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

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
    # eastern_usa = usa[usa["centroid"].x > -100]
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

    # Plot the states with county information
    states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

    # Make first cluster orange, second cluster green, third cluster red, fourth cluster purple, fifth cluster brown
    custom_colors = [
        "orange",
        "green",
        "red",
        "purple",
    ]
    cmap = mcolors.ListedColormap(custom_colors)

    # Make the colors same for same cluster and use different colors for different clusters
    cluster_plot = merged_data.plot(
        column="cluster",
        ax=ax,
        legend=True,
        cmap=cmap,  # cmap,
        linewidth=0.5,
        edgecolor="0.8",
        legend_kwds={
            "shrink": 0.6,  # Adjust the size of the colorbar
            # "label": "Clusters",  # Title for the colorbar
            "pad": 0.0001,  # Reduce the padding to bring the colorbar closer
        },
    )

    # Set the title of the plot to empty
    plt.title("")  # "Cluster Distribution of U.S. Counties"

    # Increase the fontsize of the colorbar title
    colorbar = cluster_plot.get_figure().get_axes()[-1]
    # colorbar.set_title(title, fontsize=14)
    colorbar.set_ylabel("Clusters", fontsize=18, rotation=90, labelpad=10)
    colorbar.tick_params(labelsize=18)  # Set the fontsize of the colorbar ticks
    colorbar.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Set tick labels to integers

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
    plt.savefig(
        "../figures2/cluster_distribution_5_2.png",
        dpi=500,
        bbox_inches="tight",
    )

    # Save the plot as a high-resolution image in SVG format
    plt.savefig(
        "../figures2/cluster_distribution_5_2.svg",
        dpi=500,
        bbox_inches="tight",
    )

    plt.show()


def main():
    """
    Main function to execute the steps for visualizing cluster distributions.
    """
    # Define file paths or URLs
    data_file_path = "../clustering_data/clustering_results_5.csv"
    shapefile_url = (
        "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_500k.zip"
    )
    shapefile_state_url = (
        "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_500k.zip"
    )

    # Load the data
    cluster_data = load_clustering_data(data_file_path)
    county_shapefiles = load_county_shapefiles(shapefile_url)

    # Prepare and merge the data
    prepared_data = prepare_data(county_shapefiles, cluster_data)

    # Plot the clusters
    plot_clusters(prepared_data, shapefile_url, shapefile_state_url)


# Execute the main function
if __name__ == "__main__":
    main()

# Print the message to indicate successful execution
print("Cluster visualization completed successfully.")

# END
