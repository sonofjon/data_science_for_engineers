#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the working directory for data and plots
work_dir = "/Users/andreas/Dropbox/Documents/Education/Data Science for Engineers/Assignment 1"  # Adjust this path as needed

# Build the full path to the CSV file
data_path = os.path.join(work_dir, "Auto_mpg.csv")

# Load the dataset
df = pd.read_csv(data_path)

# Optional: Inspect your dataframe
print(df.head())

# 1. Plot histograms for all numeric variables
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()  # Adjust subplot spacing
hist_output = os.path.join(work_dir, "histograms.pdf")
plt.savefig(hist_output)
plt.close()  # Close the histogram figure

# 2. Create a scatter plot matrix (pairplot)
pairplot_fig = sns.pairplot(df)
pairplot_output = os.path.join(work_dir, "pairplot.pdf")
pairplot_fig.savefig(pairplot_output)
plt.close()  # Close the pairplot figure

# 3. Produce and Export a Correlation Heatmap
corr_matrix = df.corr()  # Compute Pearson correlation coefficients
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()  # Adjust layout before saving
corr_output = os.path.join(work_dir, "correlation_heatmap.pdf")
plt.savefig(corr_output)
plt.close()  # Close the heatmap figure

print("Plots saved as:")
print(" - Histograms:    {}".format(hist_output))
print(" - Pairplot:      {}".format(pairplot_output))
print(" - Correlation Heatmap: {}".format(corr_output))
