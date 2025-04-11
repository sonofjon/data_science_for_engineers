#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (adjust the file path if needed)
df = pd.read_csv("~/Dropbox/Documents/Education/Data Science for Engineers/Assignment 1/Auto_mpg.csv")

# Optional: Inspect your dataframe
print(df.head())

# Plot histograms for all variables (numeric columns are automatically chosen)
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()   # Adjust subplot spacing to prevent overlap
plt.show()

# Create a scatter plot matrix (pairplot) to show relationships between variables
sns.pairplot(df)
plt.show()
