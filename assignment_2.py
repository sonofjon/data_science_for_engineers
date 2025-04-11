#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define work directory and ensure it exists
work_dir = "/Users/andreas/Dropbox/Documents/Education/Data Science for Engineers/Assignment 2"
os.makedirs(work_dir, exist_ok=True)

# File paths
csv_path = os.path.join(work_dir, 'Auto_mpg.csv')
boxplot_pdf_path = os.path.join(work_dir, 'normalized_boxplot.pdf')
score_pdf_path = os.path.join(work_dir, 'pca_scoreplot.pdf')
loading_pdf_path = os.path.join(work_dir, 'pca_loadingplot.pdf')
score_color_pdf_path = os.path.join(work_dir, 'pca_scoreplot_color.pdf')

# Read CSV; assume missing values are indicated by '?'
df = pd.read_csv(csv_path, na_values=['?'])
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Identify numeric columns (excludes non-numeric e.g., 'Car_name')
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Impute missing values with the median
# for col in numeric_cols:
#     df[col] = df[col].fillna(df[col].median())

# Normalize data using StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[numeric_cols])
normalized_df = pd.DataFrame(normalized_data, columns=numeric_cols)

# Apply PCA
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(normalized_data)

# ----- Create boxplot of normalized data -----
plt.figure()
normalized_df.boxplot()
plt.title("Box Plot of Normalized Numeric Features")
plt.ylabel("Standardized Value (z-score)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(boxplot_pdf_path, format="pdf")
plt.close()
print(f"Boxplot saved to: {boxplot_pdf_path}")

# ----- Create PCA score plot (PC1 vs. PC2) -----
plt.figure()
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.7, edgecolor='k')
# Compute symmetrical axis limits
limit = np.ceil(max(np.abs(pca_scores[:, 0]).max(), np.abs(pca_scores[:, 1]).max()))
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA Score Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig(score_pdf_path, format="pdf")
plt.close()
print(f"PCA score plot saved to: {score_pdf_path}")

# ----- Create PCA loading plot -----
loadings = pca.components_.T  # shape: (n_features, n_components)
plt.figure()
plt.scatter(loadings[:, 0], loadings[:, 1], alpha=0.7, edgecolor='k')

# Annotate plot markers
for i, feature in enumerate(numeric_cols):
    # Using annotate with a fixed offset in points:
    plt.annotate(feature,
                 (loadings[i, 0], loadings[i, 1]),
                 xytext=(5, 0),  # 5 points right
                 textcoords='offset points',
                 ha='left', fontsize=9)
plt.xlabel("PC1 loading")
plt.ylabel("PC2 loading")
plt.title("PCA Loading Plot")
plt.grid(True)
# Compute symmetrical axis limits
lx_min, lx_max = loadings[:, 0].min(), loadings[:, 0].max()
ly_min, ly_max = loadings[:, 1].min(), loadings[:, 1].max()
limit_loading = np.ceil(max(abs(lx_min), abs(lx_max), abs(ly_min), abs(ly_max)))
plt.xlim(-limit_loading, limit_loading)
plt.ylim(-limit_loading, limit_loading)
plt.tight_layout()
plt.savefig(loading_pdf_path, format="pdf")
plt.close()
print(f"PCA loading plot saved to: {loading_pdf_path}")

# ----- Create colored PCA score plots -----
# One subplot per variable (excluding 'Car_name')
plot_vars = [col for col in df.columns if col != 'Car_name']
n_vars = len(plot_vars)
# Determine grid layout (roughly square)
cols_subplot = math.ceil(math.sqrt(n_vars))
rows_subplot = math.ceil(n_vars / cols_subplot)

fig, axes = plt.subplots(rows_subplot, cols_subplot, figsize=(cols_subplot * 4, rows_subplot * 4), squeeze=False)
axes = axes.flatten()

for i, var in enumerate(plot_vars):
    ax = axes[i]
    # Variable is numeric and continuous
    if df[var].dtype.kind in 'biufc' and df[var].nunique() > 10:
        sc = ax.scatter(pca_scores[:, 0], pca_scores[:, 1],
                        c=df[var],
                        cmap='viridis',
                        alpha=0.7, edgecolor='k')
        cbar = fig.colorbar(sc, ax=ax)
        ax.set_title(f"{var} (continuous)")
    else:
        # Variable is categorical
        uniques = sorted(df[var].unique())
        colors = {val: plt.cm.tab10(j % 10) for j, val in enumerate(uniques)}
        for val in uniques:
            idx = df[df[var] == val].index
            ax.scatter(pca_scores[idx, 0], pca_scores[idx, 1],
                       color=colors[val],
                       alpha=0.7, edgecolor='k', label=str(val))
        ax.legend(title=var, fontsize='x-small', loc='best')
        ax.set_title(f"{var} (categorical)")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(True)

# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(score_color_pdf_path, format="pdf")
plt.close()
print(f"PCA colored score plot saved to: {score_color_pdf_path}")
