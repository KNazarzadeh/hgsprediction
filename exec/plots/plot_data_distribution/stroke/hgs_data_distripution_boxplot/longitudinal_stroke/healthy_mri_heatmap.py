import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_results import healthy
import statsmodels.api as sm

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

df_2 = healthy.load_hgs_predicted_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session="2",
)

df_3 = healthy.load_hgs_predicted_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session="3",
)

target_string = " ".join([word.upper() for word in target.split("_")])
df_1_scan = df_2[[f"1st_scan_{target}_actual", "1st_scan_age", "1st_scan_bmi", "1st_scan_height", "1st_scan_waist_to_hip_ratio"]]
df_2_scan = df_3[[f"2nd_scan_{target}_actual", "2nd_scan_age", "2nd_scan_bmi", "2nd_scan_height", "2nd_scan_waist_to_hip_ratio"]]

prefix = f"1st_scan_"
df_1_scan.columns = df_1_scan.columns.str.replace(prefix, '')
df_1_scan = df_1_scan.rename(columns={"waist_to_hip_ratio": "WHR", f"{target}_actual": "Actual HGS", "bmi":"BMI", "age":"Age", "height":"Height"})
prefix = f"2nd_scan_"
df_2_scan.columns = df_2_scan.columns.str.replace(prefix, '')
# Assuming 'target' contains the column name you want to rename
df_2_scan = df_2_scan.rename(columns={"waist_to_hip_ratio": "WHR", f"{target}_actual": "Actual HGS", "bmi":"BMI", "age":"Age", "height":"Height"})

# Create a heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_context("poster")
ax.set_box_aspect(1)
sns.heatmap(df_1_scan.corr(), annot=True, fmt='.2f', linewidths=.5, cbar=True)
# Set labels and title
plt.xlabel('')
plt.ylabel('')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)  # Adjust the font size as needed
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)  # Adjust the font size as needed

plt.title(f'1st MRI scan - Healthy controls (N={len(df_1_scan)})\nTarget={target_string}', fontsize=18, fontweight="bold")
plt.show()
plt.savefig(f"heatmap_1st_scan_{target}.png")
plt.close()

# Create a heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_context("poster")
ax.set_box_aspect(1)
sns.heatmap(df_2_scan.corr(), annot=True, fmt='.2f', linewidths=.5, cbar=True)
# Set labels and title
plt.xlabel('')
plt.ylabel('')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)  # Adjust the font size as needed
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)  # Adjust the font size as needed
plt.title(f'2nd MRI scan - Healthy controls (N={len(df_2_scan)})\nTarget={target_string}', fontsize=18, fontweight="bold")
plt.show()
plt.savefig(f"heatmap_2nd_scan_{target}.png")
plt.close()


print("===== Done! =====")
embed(globals(), locals())
