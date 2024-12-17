"""
This script creates a bar plot showing the percentage of compound events for each state.

Author: Javed Ali
Email: javedali28@gmail.com
Date: July 25, 2024
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv("../final_df_95th_coastal.csv")

# Group the data by state and sum the relevant columns
# state_data = (
#     df.groupby("state")
#     .agg({"total_events": "sum", "number_of_compound_events": "sum"})
#     .reset_index()
# )

# Group the data by state and sum the relevant columns
state_data = (
    df.groupby("state")
    .agg(
        {
            "total_events": "sum",
            "number_of_compound_events": "sum",
            "compound_loss_property_damage": "sum",
            "compound_loss_crop_damage": "sum",
            "compound_loss_injuries": "sum",
            "compound_loss_fatalities": "sum",
        }
    )
    .reset_index()
)

# Calculate the percentage of compound events for each state
state_data["compound_event_percentage"] = (
    state_data["number_of_compound_events"] / state_data["total_events"]
) * 100

# Sort the data by percentage in descending order
state_data_sorted = state_data.sort_values("compound_event_percentage", ascending=False)

# Set up the plot style
plt.style.use("default")
plt.figure(figsize=(12, 6))

# Create the bar plot
sns.barplot(
    x="state",
    y="compound_event_percentage",
    data=state_data_sorted,
    hue="compound_event_percentage",
    palette="rocket_r",
    legend=False,
)

# Customize the plot
plt.title("Percentage of Compound Events by State", fontsize=18)
plt.xlabel("State", fontsize=16)
plt.ylabel("Compound Events (%)", fontsize=16)
plt.xticks(rotation=0, ha="center", fontsize=15)
plt.yticks(fontsize=15)

# Hide the top and right spines
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Add percentage labels on top of each bar
for i, v in enumerate(state_data_sorted["compound_event_percentage"]):
    plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("../figures2/compound_events_by_state.png", dpi=400)

# Save the plot as a SVG file for vector graphics
# plt.savefig("../figures/compound_events_by_state.svg", dpi=600)

plt.close()


def format_label(value):
    """
    Format the label value into K, M, B for thousands, millions, and billions.
    """
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.1f}"


# List of loss types
loss_types = [
    "compound_loss_property_damage",
    "compound_loss_crop_damage",
    "compound_loss_injuries",
    "compound_loss_fatalities",
]

# Create bar plots for each type of loss
for loss_type in loss_types:
    # Sort the data by the current loss type in descending order
    state_data_sorted = state_data.sort_values(loss_type, ascending=False)

    # Set up the plot style
    plt.figure(figsize=(12, 6))

    # Create the bar plot
    sns.barplot(
        x="state",
        y=loss_type,
        data=state_data_sorted,
        hue=loss_type,
        palette="rocket_r",
        legend=False,
    )

    # Customize the plot
    plt.title(f"Total {loss_type.replace('_', ' ').title()} by State", fontsize=18)
    plt.xlabel("State", fontsize=16)
    plt.ylabel(f"{loss_type.replace('_', ' ').title()}", fontsize=16)
    plt.xticks(rotation=0, ha="center", fontsize=15)
    plt.yticks(fontsize=15)

    # Hide the top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Add labels on top of each bar
    for i, v in enumerate(state_data_sorted[loss_type]):
        plt.text(i, v, format_label(v), ha="center", va="bottom")

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f"../figures2/{loss_type}_by_state.png", dpi=400)

    # Save the plot as a SVG file for vector graphics
    # plt.savefig(f"../figures/{loss_type}_by_state.svg", dpi=600)

    plt.close()
