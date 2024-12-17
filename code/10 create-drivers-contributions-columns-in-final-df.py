"""
Create columns for contributions percentage of hydrometeorological drivers to compound flooding events

Author: Javed Ali
Email: javed.ali@ucf.edu
Date: January 4th, 2024

Description:
    This script creates columns for contributions percentage of different hydrometeorological drivers to Compound Flooding Events.
    The hydrometeorological drivers are:
        1. Storm Surge
        2. Precipitation
        3. River Discharge
        4. Waves
        5. Soil Moisture

Usage:
    1. Make sure the CSV file containing the data is located in the specified data_path.
    2. Run the script to calculate the percentage contribution of each hydrometeorological driver to Compound Flooding Events.
    3. The script will create new columns in the dataframe with the percentage contribution values.
    4. The dataframe will be saved to a CSV file named "final_df_with_percentage_contribution.csv".
    5. The script will print the updated dataframe and a success message.

Dependencies:
    - pandas
    - numpy
    - tqdm

"""

# Import libraries
import pandas as pd
from tqdm import tqdm


def calculate_percentage_contribution(df, driver_column, event_column):
    """
    Calculate the percentage contribution of a hydrometeorological driver to compound flooding events.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        driver_column (str): The column name of the driver in the dataframe.
        event_column (str): The column name of the number of compound events in the dataframe.

    Returns:
        pandas.Series: The series containing the percentage contribution values.

    """
    return (df[driver_column] / df[event_column]) * 100


# Path to the folder containing CSV file
data_path = "../final_df.csv"

# Read the CSV file
df = pd.read_csv(data_path)

# Create a list of columns to be used for calculating the percentage contribution
driver_columns = [
    "frequency_count_Percentiles_precip",
    "frequency_count_Percentiles_discharge",
    "frequency_count_Percentiles_soil_moisture",
    "frequency_count_surge_percentiles",
    "frequency_count_waveHs_percentiles",
]

# Calculate the percentage contribution for each driver
for driver_column in tqdm(driver_columns):
    if driver_column == "frequency_count_surge_percentiles":
        percentage_column = "percentage_contribution_surge"
    elif driver_column == "frequency_count_waveHs_percentiles":
        percentage_column = "percentage_contribution_waveHs"
    else:
        percentage_column = f"percentage_contribution_{driver_column.split('_')[-1]}"

    df[percentage_column] = calculate_percentage_contribution(
        df, driver_column, "number_of_compound_events"
    )


# Save the dataframe to a CSV file
df.to_csv("../final_df_with_percentage_contribution.csv", index=False)

# Print the dataframe
print(df)

# Print the status of the script
print("Script completed successfully!")

# End of script
