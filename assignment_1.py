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

# Plot histograms for all numeric variables
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()  # Adjust subplot spacing
# Save the histograms figure as a PDF within the work directory
hist_output = os.path.join(work_dir, "histograms.pdf")
plt.savefig(hist_output)
plt.close()  # Close the figure

# Create a scatter plot matrix (pairplot) to show relationships among variables
pairplot_fig = sns.pairplot(df)
pairplot_output = os.path.join(work_dir, "pairplot.pdf")
pairplot_fig.savefig(pairplot_output)
plt.close()  # Close the pairplot figure

print(f"Plots saved as:\n - {hist_output}\n - {pairplot_output}")
