#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def determine_symmetric_limit(x, y):
    """Determine symmetric axis limits that end on the nearest even tick number."""
    max_val = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # Round up to next whole number
    lim = np.ceil(max_val)
    # If the result is odd, add one to make it even
    if lim % 2 != 0:
        lim += 1
    return lim

# Define work directory and ensure it exists
work_dir = "/Users/andreas/Dropbox/Documents/Education/Data Science for Engineers/Assignment 2"
os.makedirs(work_dir, exist_ok=True)

# Build file paths for input CSV and output PDFs
csv_path = os.path.join(work_dir, 'Auto_mpg.csv')
boxplot_pdf_path = os.path.join(work_dir, 'normalized_boxplot.pdf')
score_pdf_path = os.path.join(work_dir, 'pca_scoreplot.pdf')
loading_pdf_path = os.path.join(work_dir, 'pca_loadingplot.pdf')

# Read CSV; assume missing values are indicated by '?'
df = pd.read_csv(csv_path, na_values=['?'])
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Identify numeric columns (exclude non-numeric, e.g., 'Car_name')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Impute missing numeric data with the median value
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Normalize the data using StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[numeric_cols])
normalized_df = pd.DataFrame(normalized_data, columns=numeric_cols)

# ----- Create and Save Boxplot of Normalized Data -----
plt.figure(figsize=(10, 6))
normalized_df.boxplot()
plt.title("Box Plot of Normalized Numeric Features")
plt.ylabel("Standardized Value (z-score)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(boxplot_pdf_path, format="pdf")
plt.close()
print(f"Boxplot saved to: {boxplot_pdf_path}")

# ----- Apply PCA (2 components) -----
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(normalized_data)

# ----- Create and Save PCA Score Plot (PC1 vs. PC2) -----
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.7, edgecolor='k')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA Score Plot")
plt.grid(True)

# Determine symmetric limits for score plot
limit_score = determine_symmetric_limit(pca_scores[:, 0], pca_scores[:, 1])
plt.xlim(-limit_score, limit_score)
plt.ylim(-limit_score, limit_score)

plt.tight_layout()
plt.savefig(score_pdf_path, format="pdf")
plt.close()
print(f"PCA score plot saved to: {score_pdf_path}")

# ----- Create and Save PCA Loading Plot -----
loadings = pca.components_.T  # shape: (n_features, n_components)
plt.figure(figsize=(8, 6))
plt.scatter(loadings[:, 0], loadings[:, 1], s=100, color='red')
for i, feature in enumerate(numeric_cols):
    plt.text(loadings[i, 0] * 1.05, loadings[i, 1] * 1.05, feature, fontsize=9)
plt.xlabel("PC1 loading")
plt.ylabel("PC2 loading")
plt.title("PCA Loading Plot")
plt.grid(True)

# Determine symmetric limits for loading plot
limit_loading = determine_symmetric_limit(loadings[:, 0], loadings[:, 1])
plt.xlim(-limit_loading, limit_loading)
plt.ylim(-limit_loading, limit_loading)

plt.tight_layout()
plt.savefig(loading_pdf_path, format="pdf")
plt.close()
print(f"PCA loading plot saved to: {loading_pdf_path}")
