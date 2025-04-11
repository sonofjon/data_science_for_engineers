#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Impute missing numeric data with the median value (avoiding chained assignment)
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

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

# Set symmetrical axis limits for score plot:
x_min, x_max = pca_scores[:, 0].min(), pca_scores[:, 0].max()
y_min, y_max = pca_scores[:, 1].min(), pca_scores[:, 1].max()
limit_x = np.ceil(max(abs(x_min), abs(x_max)))
limit_y = np.ceil(max(abs(y_min), abs(y_max)))
# Use the larger limit to have similar spacing in both axes if desired:
limit = max(limit_x, limit_y)
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)

plt.tight_layout()
plt.savefig(score_pdf_path, format="pdf")
plt.close()
print(f"PCA score plot saved to: {score_pdf_path}")

# ----- Create and Save PCA Loading Plot -----
# loadings: (n_features, n_components)
loadings = pca.components_.T
plt.figure(figsize=(8, 6))
plt.scatter(loadings[:, 0], loadings[:, 1], s=100, color='red')
for i, feature in enumerate(numeric_cols):
    plt.text(loadings[i, 0]*1.05, loadings[i, 1]*1.05, feature, fontsize=9)

plt.xlabel("PC1 loading")
plt.ylabel("PC2 loading")
plt.title("PCA Loading Plot")
plt.grid(True)

# Set symmetrical axis limits for loading plot:
lx_min, lx_max = loadings[:, 0].min(), loadings[:, 0].max()
ly_min, ly_max = loadings[:, 1].min(), loadings[:, 1].max()
limit_lx = np.ceil(max(abs(lx_min), abs(lx_max)))
limit_ly = np.ceil(max(abs(ly_min), abs(ly_max)))
limit_loading = max(limit_lx, limit_ly)
plt.xlim(-limit_loading, limit_loading)
plt.ylim(-limit_loading, limit_loading)

plt.tight_layout()
plt.savefig(loading_pdf_path, format="pdf")
plt.close()
print(f"PCA loading plot saved to: {loading_pdf_path}")
