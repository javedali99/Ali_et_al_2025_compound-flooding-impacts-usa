"""
This script visualizes the spatial distribution of various metrics related to compound flooding events
in U.S. coastal counties. It creates choropleth maps to show the percentage of compound events, loss percentages,
and percentage contributions of different drivers.

Author: Javed Ali
Email: javedali28@gmail.com
Date: May 9, 2024
"""

import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from geopandas.tools import sjoin
from matplotlib.colors import BoundaryNorm, ListedColormap


def load_data(filepath):
    """
    Loads data from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file containing the data.

    Returns:
        DataFrame: A pandas DataFrame containing the loaded data.
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


def ensure_directory_exists(file_path):
    """
    Ensure that the directory for the given file path exists.
    If it does not exist, create it.

    Args:
        file_path (str): The file path for which to ensure the directory exists.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def prepare_data(county_data, event_data):
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

    # Filter the counties based on the longitude of the centroids
    eastern_usa = merged_data[merged_data["centroid"].x > -100]

    return eastern_usa


# Define discrete levels for each specific column
colorbar_ranges = {
    "percentage_CE": [50, 60, 70, 80, 90, 100],
    "loss_percentage_compound_property_damage": [90, 92, 94, 96, 98, 100],
    "loss_percentage_compound_crop_damage": [90, 92, 94, 96, 98, 100],
    "percentage_contribution_precip": [40, 50, 60, 70, 80, 90, 100],
    "percentage_contribution_discharge": [40, 50, 60, 70, 80, 90, 100],
    "percentage_contribution_moisture": [40, 50, 60, 70, 80, 90, 100],
    "percentage_contribution_surge": [0, 20, 40, 60, 80, 100],
    "percentage_contribution_waveHs": [0, 20, 40, 60, 80, 100],
}


# Function to get discrete colormap and norm based on the range
def get_discrete_cmap_and_norm(column):
    """
    Generate a discrete colormap and normalization based on predefined levels for a given column.

    Parameters:
        column (str): The column name to retrieve levels and generate colormap.

    Returns:
        cmap (ListedColormap): Discrete colormap based on the specified levels.
        norm (BoundaryNorm): Normalization object for mapping data values to discrete levels.
    """
    # Get predefined levels for the column or use a default range
    levels = colorbar_ranges.get(column, [0, 20, 40, 60, 80, 100])
    # Generate colors using the 'Spectral' palette
    colors = sns.color_palette("rocket_r", n_colors=len(levels) - 1)
    # Create a ListedColormap and BoundaryNorm
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True)
    return cmap, norm


def plot_map(
    merged_data,
    column,
    title,
    shapefile_url,
    shapefile_state_url,
    figure_path,
    figsize=(22, 18),
):
    """
    Plot a single choropleth map for a specified column with customized discrete color bars.

    Parameters:
        merged_data (GeoDataFrame): GeoDataFrame containing spatial data and metrics.
        column (str): Column name to be visualized.
        title (str): Title of the map.
        shapefile_url (str): URL of the shapefile for county boundaries.
        shapefile_state_url (str): URL of the shapefile for state boundaries.
        figure_path (str): Directory to save the figure.
        figsize (tuple): Size of the figure (default: (22, 18)).

    Returns:
        None. Saves the plot to the specified directory.
    """

    # Determine colormap and normalization based on the column
    if column in ["number_of_compound_events", "total_events"]:
        # Calculate min and max for the column
        min_val = merged_data[column].min()
        max_val = merged_data[column].max()

        # Define levels for discrete classes
        if min_val < 20:
            levels = [0, 20] + list(range(40, int(max_val) + 20, 20))
        else:
            levels = list(range(20, int(max_val) + 20, 20))

        # Create a colormap and normalization
        colors = sns.color_palette("rocket_r", n_colors=len(levels) - 1)
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True)

        # Add a downward-pointy colorbar for underflow
        legend_kwds = {
            "shrink": 0.7,
            "label": title,
            "pad": 0.0001,
            "extend": "min",  # Downward-pointy colorbar for underflow
        }
    elif column == "percentage_CE":
        # Use the standard discrete colormap for percentage_CE
        cmap, norm = get_discrete_cmap_and_norm(column)
        legend_kwds = {
            "shrink": 0.7,
            "label": title,
            "pad": 0.0001,
        }
    else:
        # Default to a continuous colormap for other columns
        cmap, norm = get_discrete_cmap_and_norm(column)
        legend_kwds = {
            "shrink": 0.7,
            "label": title,
            "pad": 0.0001,
        }

    # Create a new figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

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

    # Plot the counties colored by the specified column
    plot = merged_data.plot(
        column=column,
        ax=ax,
        legend=True,
        cmap=cmap,
        norm=norm,
        linewidth=0.5,
        edgecolor="0.8",
        legend_kwds={"shrink": 0.7, "label": title, "pad": 0.0001},
    )

    # Set the title of the plot
    plt.title(title, loc="left", fontweight="bold", fontsize=18)

    # Increase the fontsize of the colorbar title
    colorbar = plot.get_figure().get_axes()[-1]
    colorbar.set_ylabel(title, fontsize=18, rotation=90, labelpad=10)
    colorbar.tick_params(labelsize=18)  # Set the fontsize of the colorbar ticks

    # Hide axes and spines for better visualization
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Save the plot
    plt.savefig(
        f"{figure_path}{title.replace(' ', '_').lower()}_final.png",
        dpi=500,
        bbox_inches="tight",
    )

    # Save the plot in SVG format for better quality
    plt.savefig(
        f"{figure_path}{title.replace(' ', '_').lower()}_final.svg",
        dpi=500,
        bbox_inches="tight",
    )

    plt.close()


def plot_subplots(
    merged_data,
    columns,
    titles,
    shapefile_url,
    shapefile_state_url,
    figure_path,
    figsize=(22, 18),
    rows=2,
    cols=2,
    is_losses="losses",
):
    """
    Plots multiple choropleth maps as subplots.

    Parameters:
        merged_data (GeoDataFrame): The merged GeoDataFrame containing event data and geography.
        columns (list): A list of column names to use for coloring the counties in each subplot.
        titles (list): A list of titles for each subplot.
        shapefile_url (str): The URL to the shapefile containing U.S. county boundaries.
        shapefile_state_url (str): The URL to the shapefile containing U.S. state boundaries.
        figure_path (str): The path to save the output figures.
        figsize (tuple): The figure size (default: (20, 16)).
        rows (int): The number of rows in the subplot grid (default: 2).
        cols (int): The number of columns in the subplot grid (default: 2).
        is_losses (str): A string indicating whether the subplots are for losses or contributions.

    Returns:
        None. Saves the plot to the specified directory.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

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

    # Perform a spatial join between the state and county GeoDataFrames
    states_with_counties = sjoin(
        eastern_usa_state, merged_data, how="inner", predicate="intersects"
    )

    # Reproject merged_data to the same CRS as eastern_usa
    merged_data = merged_data.to_crs(eastern_usa.crs)

    for i, ax in enumerate(axes.flatten()):
        if i < len(columns):
            column = columns[i]
            title = titles[i]

            # Generate colormap and normalization
            cmap, norm = get_discrete_cmap_and_norm(column)

            # Plot the states with county information
            states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

            # Plot the counties colored by the specified column
            plot = merged_data.plot(
                column=column,
                ax=ax,
                legend=True,
                cmap=cmap,
                norm=norm,
                linewidth=0.5,
                edgecolor="0.8",
                legend_kwds={"shrink": 0.7, "label": title, "pad": 0.0001},
            )

            # Set the title of the subplot
            ax.set_title(title, fontsize=18, fontweight="bold", loc="left")

            # Increase the fontsize of the colorbar title
            colorbar = plot.get_figure().get_axes()[-1]
            colorbar.set_ylabel(title, fontsize=18, rotation=90, labelpad=10)
            colorbar.tick_params(labelsize=18)  # Set the fontsize of the colorbar ticks

            # Hide axes and spines
            ax.spines[:].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        else:
            ax.remove()  # Remove empty subplot

    plt.tight_layout()

    # Save the plot
    plt.savefig(
        f"{figure_path}compound_flooding_metrics_subplots_{is_losses}_final.png",
        dpi=500,
        bbox_inches="tight",
    )

    # Save the plot in SVG format for better quality
    plt.savefig(
        f"{figure_path}compound_flooding_metrics_subplots_{is_losses}_final.svg",
        dpi=500,
        bbox_inches="tight",
    )

    plt.close()


def plot_loss_subplots(
    merged_data,
    shapefile_url,
    shapefile_state_url,
    figure_path,
    figsize=(22, 18),
    rows=2,
    cols=1,
):
    """
    Plots multiple choropleth maps as subplots for specified loss percentage columns,
    while graying out counties based on total losses.

    Parameters:
        merged_data (GeoDataFrame): The merged GeoDataFrame containing event data and geography.
        shapefile_url (str): The URL to the shapefile containing U.S. county boundaries.
        shapefile_state_url (str): The URL to the shapefile containing U.S. state boundaries.
        figure_path (str): The path to save the output figures.
        figsize (tuple): The figure size (default: (22, 18)).
        rows (int): The number of rows in the subplot grid (default: 2).
        cols (int): The number of columns in the subplot grid (default: 2).
    """
    percentage_columns = [
        "loss_percentage_compound_property_damage",
        "loss_percentage_compound_crop_damage",
    ]
    total_loss_columns = [
        "total_loss_property_damage",
        "total_loss_crop_damage",
    ]
    titles = [
        "Property Damage by CF (in %)",
        "Crop Damage by CF (in %)",
    ]

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

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

    # Perform a spatial join between the state and county GeoDataFrames
    states_with_counties = sjoin(
        eastern_usa_state, merged_data, how="inner", predicate="intersects"
    )

    # Reproject merged_data to the same CRS as eastern_usa
    merged_data = merged_data.to_crs(eastern_usa.crs)

    for i, (percentage_column, total_loss_column, title) in enumerate(
        zip(percentage_columns, total_loss_columns, titles)
    ):
        ax = axes[i]

        # Generate colormap and normalization
        cmap, norm = get_discrete_cmap_and_norm(percentage_column)

        # Apply gray-out conditions based on total losses
        if "total_loss_crop_damage" in total_loss_column.lower():
            merged_data["color"] = merged_data[total_loss_column].apply(
                lambda x: "gray" if x == 0 else "original"
            )

        else:
            merged_data["color"] = "original"

        # Plot the counties
        gray_data = merged_data[merged_data["color"] == "gray"]
        colored_data = merged_data[merged_data["color"] == "original"]

        # Plot the states with county information
        states_with_counties.boundary.plot(ax=ax, linewidth=0.5, color="black")

        if not gray_data.empty:
            gray_data.plot(ax=ax, color="lightgray", linewidth=0.5, edgecolor="0.8")

        # Plot the counties colored by the percentage column
        if not colored_data.empty:
            plot = colored_data.plot(
                column=percentage_column,
                ax=ax,
                legend=True,
                cmap=cmap,
                norm=norm,
                linewidth=0.5,
                edgecolor="0.8",
                legend_kwds={
                    "shrink": 0.7,
                    "label": title,
                    "pad": 0.0001,
                    "extend": "min",
                    "extendfrac": 0.1,
                },
                # colorbar range for percentage losses
                vmin=90,
                vmax=100,
            )

        # Set the title of the subplot
        ax.set_title(title, loc="left", fontweight="bold", fontsize=18)

        # Increase the fontsize of the colorbar title
        colorbar = plot.get_figure().get_axes()[-1]
        colorbar.set_ylabel(title, fontsize=18, rotation=90, labelpad=10)
        colorbar.tick_params(labelsize=18)  # Set the fontsize of the colorbar ticks

        # Hide axes and spines
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Save the plot
    plt.savefig(
        f"{figure_path}CF_loss_percentage_subplots_final.png",
        dpi=500,
        bbox_inches="tight",
    )

    # Save the plot in SVG format for better quality
    plt.savefig(
        f"{figure_path}CF_loss_percentage_subplots_final.svg",
        dpi=500,
        bbox_inches="tight",
    )

    plt.close()


def main():
    """
    Main function to execute the steps for visualizing the spatial distribution of compound flooding metrics.
    """
    # Define file paths or URLs
    data_file_path = "../final_df_99th_coastal.csv"
    shapefile_url = (
        "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_500k.zip"
    )

    shapefile_state_url = (
        "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_500k.zip"
    )

    figure_path = "../figures2/final_maps_revision/99th/"

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # Load the data
    event_data = load_data(data_file_path)
    county_shapefiles = load_county_shapefiles(shapefile_url)

    # Prepare and merge the data
    prepared_data = prepare_data(county_shapefiles, event_data)

    # Plot the percentage of compound events
    plot_map(
        prepared_data,
        "percentage_CE",
        "Compound Flooding Events (in %)",
        shapefile_url,
        shapefile_state_url,
        figure_path=figure_path,
    )

    # Plot the total number of compound events
    plot_map(
        prepared_data,
        "number_of_compound_events",
        "Total Compound Flooding Events",
        shapefile_url,
        shapefile_state_url,
        figure_path=figure_path,
    )

    # Plot the total number of flood events
    plot_map(
        prepared_data,
        "total_events",
        "Total Flooding Events",
        shapefile_url,
        shapefile_state_url,
        figure_path=figure_path,
    )

    # Plot the loss percentages
    loss_columns = [
        "loss_percentage_compound_property_damage",
        "loss_percentage_compound_crop_damage",
        "loss_percentage_compound_injuries",
        "loss_percentage_compound_fatalities",
    ]
    loss_titles = [
        "Property Damage by CF (in %)",
        "Crop Damage by CF (in %)",
        "Injuries by CF (in %)",
        "Fatalities by CF (in %)",
    ]
    plot_subplots(
        prepared_data,
        loss_columns,
        loss_titles,
        shapefile_url,
        shapefile_state_url,
        figure_path=figure_path,
    )

    # Plot the percentage contributions of drivers
    driver_columns = [
        "percentage_contribution_precip",
        "percentage_contribution_discharge",
        "percentage_contribution_moisture",
        "percentage_contribution_surge",
        "percentage_contribution_waveHs",
    ]
    driver_titles = [
        "Precipitation Contribution to CF (in %)",
        "Discharge Contribution to CF (in %)",
        "Soil Moisture Contribution to CF (in %)",
        "Surge Contribution to CF (in %)",
        "Wave Height Contribution to CF (in %)",
    ]
    plot_subplots(
        prepared_data,
        driver_columns,
        driver_titles,
        shapefile_url,
        shapefile_state_url,
        rows=3,
        cols=2,
        figsize=(16, 18),
        is_losses="contributions",
        figure_path=figure_path,
    )

    # Plot the total losses
    plot_loss_subplots(
        prepared_data,
        shapefile_url,
        shapefile_state_url,
        figsize=(22, 18),
        figure_path=figure_path,
    )


# Execute the main function
if __name__ == "__main__":
    main()

# Print the message to indicate successful execution
print("Spatial distribution visualization completed successfully.")
