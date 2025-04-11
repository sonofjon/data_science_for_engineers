#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the working directory for data and plots
work_dir = "/Users/andreas/Dropbox/Documents/Education/Data Science for Engineers/Assignment 1"

# Build the full path to the CSV file
data_path = os.path.join(work_dir, "Auto_mpg.csv")

# Load the dataset
df = pd.read_csv(data_path)

# Optional: Inspect the dataframe
print(df.head())

# 1. Plot histograms for all variables
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()  # Adjust subplot spacing
hist_output = os.path.join(work_dir, "histograms.pdf")
plt.savefig(hist_output)
plt.close()  # Close the figure

# 2. Create a scatter plot matrix (pairplot)
pairplot_fig = sns.pairplot(df)
pairplot_output = os.path.join(work_dir, "pairplot.pdf")
pairplot_fig.savefig(pairplot_output)
plt.close()  # Close the figure

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# 3. Produce and Export a Correlation Heatmap
corr_matrix = numeric_df.corr()  # Compute Pearson correlation coefficients
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()  # Adjust layout before saving
corr_output = os.path.join(work_dir, "correlation_heatmap.pdf")
plt.savefig(corr_output)
plt.close()  # Close the figure

# 4. Descriptive Statistics
descriptive_stats = numeric_df.describe()
num_vars = len(numeric_df.columns)
num_cols = 3
num_rows = int(np.ceil(num_vars / num_cols))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))

for i, var in enumerate(numeric_df.columns):
    row_index = i // num_cols
    col_index = i % num_cols

    ax = axes[row_index, col_index] if num_rows > 1 else axes[col_index]  # Handle single-row case

    mean_val = descriptive_stats.loc['mean', var]
    std_val = descriptive_stats.loc['std', var]

    ax.errorbar(mean_val, 0, xerr=std_val, fmt='o', color='blue', capsize=5, label='Mean ± Std Dev')
    ax.axvline(descriptive_stats.loc['50%', var], color='green', linestyle='--', label='Median')
    ax.set_xlim(mean_val - 3 * std_val, mean_val + 3 * std_val)

    ax.set_title(f'{var} - Mean, Median, Std Dev')
    ax.set_xlabel('Value')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True)

# Remove empty subplots if the number of variables isn't a multiple of 3
if num_vars % num_cols != 0:
    for i in range(num_vars % num_cols, num_cols):
        fig.delaxes(axes[num_rows - 1, i]) if num_rows > 1 else fig.delaxes(axes[i])

plt.tight_layout()
descriptive_subplots_output = os.path.join(work_dir, "descriptive_stats.pdf")
plt.savefig(descriptive_subplots_output)
plt.close()

print("Plots saved as:")
print(" - Histograms:          {}".format(hist_output))
print(" - Pairplot:            {}".format(pairplot_output))
print(" - Correlation Heatmap: {}".format(corr_output))
print(" - Descriptive Stats:   {}".format(descriptive_subplots_output))
