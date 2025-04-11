#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define work directory and ensure it exists
work_dir = "/Users/andreas/Dropbox/Documents/Education/Data Science for Engineers/Assignment 2"
os.makedirs(work_dir, exist_ok=True)

# Build file paths for input CSV and output PDF
csv_path = os.path.join(work_dir, 'Auto_mpg.csv')
pdf_path = os.path.join(work_dir, 'normalized_boxplot.pdf')

# Read CSV; assume missing values are indicated by '?'
df = pd.read_csv(csv_path, na_values=['?'])

# Print missing data summary
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Identify numeric columns (excluding non-numeric, e.g., 'Car_name')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Impute missing numeric data with the median value without chained assignment
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Normalize the data using StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[numeric_cols])
normalized_df = pd.DataFrame(normalized_data, columns=numeric_cols)

# Create a box plot of the normalized numeric data
plt.figure(figsize=(10, 6))
normalized_df.boxplot()
plt.title("Box Plot of Normalized Numeric Features")
plt.ylabel("Standardized Value (z-score)")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PDF into work_dir
plt.savefig(pdf_path, format="pdf")
plt.close()

print(f"Plot saved to: {pdf_path}")
