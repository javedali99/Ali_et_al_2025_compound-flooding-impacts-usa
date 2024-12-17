"""
Subset CSV files

Author: Javed Ali
Email: javed.ali@ucf.edu
Date: January 2, 2024

This script reads a CSV file, subsets the data based on a threshold value,
and saves the subsetted data to a new CSV file.
"""

# Import libraries
import pandas as pd


def subset_csv(data_path, threshold, output_path):
    """
    Subset the data in a CSV file based on a threshold value.

    Args:
        data_path (str): Path to the CSV file.
        threshold (float): Threshold value for subsetting the data.
        output_path (str): Path to save the subsetted data.

    Returns:
        None
    """
    # Read CSV file
    df = pd.read_csv(data_path)

    # Subset the data based on the threshold value
    df = df[df["threshold"] == threshold]

    # Make all columns limit to 2 decimal places
    df = df.round(2)

    # Save the subsetted data to CSV file
    df.to_csv(output_path, index=False)

    return df


# Path to the folder containing CSV file
data_path = "../final_df_with_percentage_contribution.csv"

# Set the threshold value
thresholds = [0.99, 0.95, 0.90]


# Exclude rows with certain FIPS codes
excluded_fips = [
    48469,
    48239,
    48361,
    22019,
    22053,
    22099,
    22007,
    22093,
    22005,
    22095,
    22089,
    22063,
    22105,
    12077,
    12107,
    12019,
    13049,
    13025,
    13305,
    45049,
    45035,
    45015,
    37103,
    37147,
    51041,
    51087,
    51085,
    51101,
    51097,
    51033,
    51177,
    24027,
    36071,
    36111,
    36027,
    36079,
]

# Subset the CSV file based on the threshold values and save to respective files
for threshold in thresholds:
    output_path = f"../final_df_{int(threshold * 100)}th.csv"
    df_threshold = subset_csv(data_path, threshold, output_path)
    print(f"Subsetted data for threshold {threshold} saved to {output_path}")

    # Exclude the specified FIPS codes from the data for coastal counties
    df_coastal_counties = df_threshold[~df_threshold["FIPS"].isin(excluded_fips)]

    # Save the coastal counties data to a new CSV file
    output_path_coastal = f"../final_df_{int(threshold * 100)}th_coastal.csv"
    df_coastal_counties.to_csv(output_path_coastal, index=False)

    # Print success message
    print(
        f"Coastal counties data for threshold {threshold} saved to {output_path_coastal}"
    )

# End of script
