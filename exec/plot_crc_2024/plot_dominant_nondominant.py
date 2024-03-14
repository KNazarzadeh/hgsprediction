

import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from hgsprediction.load_results.load_trained_scores_results import load_scores_trained, load_test_scores_trained

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Inputs : Required inputs
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
confound_status = sys.argv[4]
model_name = sys.argv[5]
n_repeats = sys.argv[6]
n_folds = sys.argv[7]

df_female_dominant = load_scores_trained(
    population,
    mri_status,
    confound_status,
    "female",
    feature_type,
    "hgs_dominant",
    model_name,
    n_repeats,
    n_folds,
)

df_male_dominant = load_scores_trained(
    population,
    mri_status,
    confound_status,
    "male",
    feature_type,
    "hgs_dominant",
    model_name,
    n_repeats,
    n_folds,
)

df_female_dominant.loc[:, "gender"] = "Female"
df_male_dominant.loc[:, "gender"] = "Male"

df_dominant = pd.concat([df_female_dominant, df_male_dominant], axis=0)
df_dominant.loc[:, "hgs_target"] = "Dominant"

###############################################################################
df_female_nondominant = load_scores_trained(
    population,
    mri_status,
    confound_status,
    "female",
    feature_type,
    "hgs_nondominant",
    model_name,
    n_repeats,
    n_folds,
)

df_male_nondominant = load_scores_trained(
    population,
    mri_status,
    confound_status,
    "male",
    feature_type,
    "hgs_nondominant",
    model_name,
    n_repeats,
    n_folds,
)

df_female_nondominant.loc[:, "gender"] = "Female"
df_male_nondominant.loc[:, "gender"] = "Male"

df_nondominant = pd.concat([df_female_nondominant, df_male_nondominant], axis=0)
df_nondominant.loc[:, "hgs_target"] = "Non-Dominant"

###############################################################################

df = pd.concat([df_dominant, df_nondominant], axis=0)

###############################################################################
# Create a figure with the desired size
palette_colors = sns.color_palette("Paired")
custome_palette = {"Dominant": palette_colors[1], "Non-Dominant":palette_colors[0]}
fig = plt.figure(figsize=(15, 10))

sns.set_style("whitegrid")
# Plot the violin plot
sns.violinplot(data=df, x="hgs_target", y="test_pearson_corr", hue="hgs_target", palette=custome_palette, linewidth=3, inner="box")

# Set xlabel, ylabel, and title
plt.xlabel("")
plt.ylabel("r value", fontsize=50, fontweight="bold")

# Change x-axis tick labels
# new_xticklabels = ["Dominant", "Non-Dominant"]  # Replace with your desired labels
# plt.xticks(ticks=[0, 1], labels=new_xticklabels, fontsize=30, weight='bold')
plt.xticks(fontsize=20, weight='bold')

ymin, ymax = plt.ylim()
y_step_value = 0.02
plt.yticks(np.arange(round(ymin/0.01)*.01, round(ymax/0.01)*.01+.03, y_step_value), fontsize=20, weight='bold', y=1.01)

plt.title("Predicting HGS from anthropometric features", fontsize=20, fontweight="bold", y=1.03)

# Place legend outside the plot
# plt.legend(title="Samples", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')
# Show the plot
plt.show()
# Save the plot to a file
plt.savefig("dominant_nondominant.png")
# Close the figure
plt.close()


###############################################################################
# Create a figure with the desired size
# palette_male = sns.color_palette("Paired")
# palette_female = sns.color_palette("PiYG")
# custom_color = {'Female': palette_female[0], 'Male': palette_male[1]}

custom_color = {'Female': "#f45f74", 'Male': "#00b0be"}

fig = plt.figure(figsize=(15, 10))

sns.set_style("whitegrid")
# Plot the violin plot
sns.violinplot(data=df, x="hgs_target", y="test_pearson_corr", hue="gender", palette=custom_color, linewidth=3, inner="box")

# Set xlabel, ylabel, and title
plt.xlabel("")
plt.ylabel("r value", fontsize=50, fontweight="bold")
plt.xticks(fontsize=20, weight='bold')

ymin, ymax = plt.ylim()
y_step_value = 0.02
# plt.yticks(np.arange(round(ymin/0.01)*.01-0, round(ymax/0.01)*.01+.03, y_step_value), fontsize=20, weight='bold', y=1.01)
plt.yticks(np.arange(0.26, round(ymax/0.01)*.01+.03, y_step_value), fontsize=20, weight='bold', y=1.01)

plt.title("Predicting HGS from anthropometric features", fontsize=20, fontweight="bold", y=1.03)

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.001, 1), loc='upper left')
# Show the plot
plt.show()
# Save the plot to a file
plt.savefig("dominant_nondominant_by_gender.png")
# Close the figure
plt.close()

print("===== Done! =====")
embed(globals(), locals())

