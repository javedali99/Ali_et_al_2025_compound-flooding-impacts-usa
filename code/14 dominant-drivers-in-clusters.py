"""
Calculate Average Contribution Percentages for Each Variable per Cluster and Create Pie Charts

Author: Javed Ali
Email: javedali28@gmail.com
Date: May 13, 2024

Description: 
This script calculates the average contribution percentage for each variable per cluster and creates pie charts 
for each cluster showing the average contribution percentages for each variable. The script loads the dataframes, 
merges them based on 'county_name' and 'FIPS' columns, calculates the average contribution percentages for each 
variable per cluster, and creates pie charts for each cluster.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def merge_dataframes(df1, df2):
    """
    Merge two dataframes based on 'county_name' and 'FIPS' columns.

    Args:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    merged_df = pd.merge(df1, df2, on=["county_name", "FIPS"], how="left")

    return merged_df


def calculate_cluster_variable_averages(df):
    """
    Calculate the average contribution percentage for each variable per cluster.

    Args:
        df (pd.DataFrame): The merged dataframe.

    Returns:
        pd.DataFrame: A dataframe with the average contribution percentages
                      for each variable per cluster.
    """
    cluster_cols = [
        "cluster",
        "percentage_contribution_precip",
        "percentage_contribution_discharge",
        "percentage_contribution_moisture",
        "percentage_contribution_surge",
        "percentage_contribution_waveHs",
    ]

    cluster_variable_averages = (
        df[cluster_cols]
        .groupby("cluster")
        .agg(
            precip_avg=("percentage_contribution_precip", "mean"),
            discharge_avg=("percentage_contribution_discharge", "mean"),
            moisture_avg=("percentage_contribution_moisture", "mean"),
            surge_avg=("percentage_contribution_surge", "mean"),
            waveHs_avg=("percentage_contribution_waveHs", "mean"),
        )
    )

    # Print the average contribution percentages for each variable per cluster
    print(f"Average Contribution Percentages for Each Variable per Cluster:")
    print("precip_avg: ", cluster_variable_averages["precip_avg"])
    print("discharge_avg: ", cluster_variable_averages["discharge_avg"])
    print("moisture_avg: ", cluster_variable_averages["moisture_avg"])
    print("surge_avg: ", cluster_variable_averages["surge_avg"])
    print("waveHs_avg: ", cluster_variable_averages["waveHs_avg"])

    return cluster_variable_averages


def create_pie_charts(df, save_path):
    """
    Create pie charts for each cluster showing the average contribution percentages
    for each variable.

    Args:
        df (pd.DataFrame): The dataframe with the average contribution percentages
                           for each variable per cluster.
        save_path (str): The path to save the pie chart images.
    """
    num_clusters = len(df)

    num_cols = 2
    # num_rows = 2
    num_rows = (num_clusters + 1) // num_cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_clusters, 14))
    axs = axs.flatten()

    cluster_info = {
        1: "(110 counties)",
        2: "(40 counties)",
        3: "(15 counties)",
        4: "(38 counties)",
    }

    for i, (cluster, row) in enumerate(df.iterrows()):
        values = row.values
        labels = [
            "Precipitation",
            "River Discharge",
            "Soil Moisture",
            "Storm Surge",
            "Waves",
        ]

        # Print the values and labels for each cluster
        print(f"Cluster {cluster}:")
        for label, value in zip(labels, values):
            print(f"{label}: {value:.2f}%")

        # Define the colormap and select specific colors
        cmap = plt.get_cmap("tab20c")
        colors = cmap(np.array([14, 6, 0, 9, 4]))

        patches, texts, autotexts = axs[i].pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 16},
        )

        title = f"Cluster {cluster}"
        subtitle = cluster_info.get(cluster, "")

        axs[i].set_title(
            f"{title}",
            fontsize=24,
            weight="bold",
            y=1.05,
        )

        # Add subtitle separately with different styling
        axs[i].text(
            0.5,
            1.01,
            subtitle,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[i].transAxes,
            fontsize=19,  # Subtitle fontsize
            weight="normal",  # Subtitle not bold
        )

        # Adjust the position of each label
        for text in texts:
            text.set_position((text.get_position()[0], text.get_position()[1] * 1.05))

    plt.tight_layout()

    # Save the figure
    plt.savefig(
        f"{save_path}/cluster_variable_pie_charts_5_with_counties.png",
        dpi=500,
        bbox_inches="tight",
    )

    # Save the plot in SVG format for better quality
    plt.savefig(
        f"{save_path}/cluster_variable_pie_charts_5_with_counties.svg",
        dpi=500,
        bbox_inches="tight",
    )

    plt.close()


# Load the dataframes
final_df_95th_coastal = pd.read_csv("../final_df_95th_coastal.csv")
clustering_results_6 = pd.read_csv("../clustering_data/clustering_results_5.csv")

# Merge the dataframes
merged_df = merge_dataframes(final_df_95th_coastal, clustering_results_6)

# Save the merged dataframe to a CSV file
merged_df.to_csv("../merged_df_with_clusters_5.csv", index=False)

# Calculate average contribution percentage for each variable per cluster
cluster_variable_avgs = calculate_cluster_variable_averages(merged_df)

# Create pie charts for each cluster
create_pie_charts(cluster_variable_avgs, "../figures2")

# Print the message after the script has finished running
print("Cluster variable analysis completed successfully.")

# END
