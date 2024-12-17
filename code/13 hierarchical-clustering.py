"""
This script performs hierarchical clustering on a dataset of compound flooding impacts along the USA coastline.
The dataset is read from a CSV file, preprocessed by selecting relevant columns and scaling the features,
and then hierarchical clustering is performed on the preprocessed data. The resulting dendrogram is plotted,
and a choropleth map of the clusters and subclusters is created.

Author: Javed Ali
Email: javedali28@gmail.com
Date: May 7, 2024

"""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    linkage,
    set_link_color_palette,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def read_data(file_path):
    """
    Read the CSV file and return a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file_path)

    return df


def preprocess_data(df):
    """
    Preprocess the data by selecting relevant columns and scaling the features.

    Args:
        df (pandas.DataFrame): DataFrame containing the original data.

    Returns:
        pandas.DataFrame: DataFrame with selected columns and scaled features.
    """
    # Select relevant columns for clustering
    columns = [
        "county_name",
        "FIPS",
        "state",
        # "total_events",
        # "number_of_compound_events",
        # "percentage_CE",
        "percentage_contribution_precip",
        "percentage_contribution_discharge",
        "percentage_contribution_moisture",
        "percentage_contribution_surge",
        "percentage_contribution_waveHs",
    ]
    df_selected = df[columns]

    # Remove rows where "percentage_contribution_surge" is zero
    # df_selected = df_selected[df_selected["percentage_contribution_surge"] != 0]

    # Scale the features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_selected.iloc[:, 3:]), columns=df_selected.columns[3:]
    )
    df_scaled = pd.concat([df_selected.iloc[:, :3], df_scaled], axis=1)

    return df_scaled


def perform_hierarchical_clustering(df):
    """
    Perform hierarchical clustering on the preprocessed data.

    Args:
        df (pandas.DataFrame): DataFrame with preprocessed data.

    Returns:
        numpy.ndarray: Linkage matrix representing the hierarchical clustering.
    """
    # Extract the features for clustering
    features = df.iloc[:, 3:].values

    # Perform hierarchical clustering
    linkage_matrix = linkage(
        features,
        method="ward",
        metric="euclidean",
    )  # Using Ward's method for linkage clustering

    return linkage_matrix


def plot_all_dendrogram(
    linkage_matrix,
    df,
):
    """
    Plot the dendrogram representing the hierarchical clustering.

    Args:
        linkage_matrix (numpy.ndarray): Linkage matrix representing the hierarchical clustering.
        df (pandas.DataFrame): DataFrame with preprocessed data.
    """

    df["label"] = df["county_name"] + " " + df["state"]

    # Remove "_" from the county names for better readability
    df["label"] = df["label"].str.replace("_", " ")

    plt.figure(figsize=(28, 10))

    # Define a custom color palette
    custom_palette = [
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        # "blue",
    ]

    # Make custom color pallete for the dendrogram using #66545e, #a39193, #aa6f73, #eea990, #f6e0b5
    # custom_palette = [
    #     "#66545e",
    #     "#a39193",
    #     "#aa6f73",
    #     "#eea990",
    #     "#f6e0b5",
    # ]

    # Set the color palette
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.set_link_color_palette.html#scipy.cluster.hierarchy.set_link_color_palette
    set_link_color_palette(custom_palette)

    dend = dendrogram(
        linkage_matrix,
        # labels=df.iloc[:, 0].values,
        labels=df["label"].values,
        leaf_rotation=90.0,
        leaf_font_size=8.0,
        color_threshold=10,  # Add this line to color different subclusters
        show_contracted=True,  # Show a summarized view for large clusters
        above_threshold_color="black",  # Color the clusters above the threshold in black
    )

    # Color the leaves of the dendrogram based on the cluster assignments
    for leaf, leaf_color in zip(plt.gca().get_xticklabels(), dend["leaves_color_list"]):
        leaf.set_color(leaf_color)

    # Print the color list for the leaves of the dendrogram
    # print("Color list: /n", dend["color_list"])

    plt.title("Hierarchical Clustering Dendrogram", fontsize=18)
    plt.xlabel(None)
    plt.ylabel("Distance \n(Ward's Linkage)", fontsize=14, labelpad=20)

    # Make spine invisible
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    # Hide y-axis ticks
    # plt.yticks([])

    # Increase the font size of the y-axis ticks for better readability
    plt.yticks(fontsize=14)

    plt.tight_layout()

    # Save the figure
    plt.savefig("../figures2/dendrogram_5_clusters.png", dpi=500)

    # Save the plot in SVG format for better quality
    plt.savefig("../figures2/dendrogram_5_clusters.svg", dpi=500)

    plt.close()


def main():
    # Read the data from the CSV file
    file_path = "../final_df_95th_coastal.csv"
    df = read_data(file_path)

    # Preprocess the data
    df_preprocessed = preprocess_data(df)

    # Remove rows with missing values
    # df_preprocessed = df_preprocessed.dropna()

    # Perform hierarchical clustering
    linkage_matrix = perform_hierarchical_clustering(df_preprocessed)

    # Assign counties to clusters based on a fixed number of clusters
    num_clusters = 5  # or
    distance_threshold = 10  # Adjust this value as needed
    cluster_assignments = fcluster(
        linkage_matrix, distance_threshold, criterion="distance"
    )

    # Assign counties to clusters based on a distance threshold
    # distance_threshold = 2.5  # Adjust this value as needed
    # cluster_assignments = fcluster(
    #     linkage_matrix, distance_threshold, criterion="distance"
    # )

    # Create a DataFrame with clustering results
    clustering_results = pd.DataFrame(
        {
            "county_name": df_preprocessed["county_name"],
            "FIPS": df_preprocessed["FIPS"],
            "state": df_preprocessed["state"],
            "cluster": cluster_assignments,
        }
    )

    # Save the clustering results to a CSV file
    clustering_results.to_csv("../clustering_data/clustering_results_5.csv", index=False)

    # Plot the dendrograms with 20 clusters per figure
    plot_all_dendrogram(linkage_matrix, df_preprocessed)


if __name__ == "__main__":
    main()

# Print the message after the script has finished running
print("Hierarchical clustering completed successfully.")
