
import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from hgsprediction.load_results.load_multi_samples_trained_models_results import load_scores_trained
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Inputs : Required inputs
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
confound_status = sys.argv[5]
model_name = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]

###############################################################################
###############################################################################
samplesize_list = ["10_percent", "20_percent", "40_percent", "60_percent", "80_percent", "100_percent"]
df = pd.DataFrame()
for samplesize in samplesize_list:
    print(samplesize)
    df_female = load_scores_trained(
        population,
        mri_status,
        confound_status,
        "female",
        feature_type,
        target,
        model_name,
        n_repeats,
        n_folds,
        samplesize,
    )

    df_male = load_scores_trained(
        population,
        mri_status,
        confound_status,
        "male",
        feature_type,
        target,
        model_name,
        n_repeats,
        n_folds,
        samplesize,
    )

    df_female.loc[:, "gender"] = "Female"
    df_male.loc[:, "gender"] = "Male"
    df_female.loc[:, 'samplesize_percent'] = samplesize
    df_male.loc[:, 'samplesize_percent'] = samplesize

    df_tmp = pd.concat([df_female, df_male], axis=0)
    df = pd.concat([df, df_tmp], axis=0)
    


##############################################################################
# Create a custom color palette dictionary
# Define custom palettes
# palette_male = sns.color_palette("Paired")
# palette_female = sns.color_palette("PiYG")
# custom_palette = {'Female': palette_female[1], 'Male': palette_male[1]}

custom_color = {'Female': "#f45f74", 'Male': "#00b0be"}


fig = plt.figure(figsize=(15, 10))

sns.set_style("whitegrid")
# Plot the violin plot
sns.violinplot(data=df, x="samplesize_percent", y="test_pearson_corr", hue="gender", palette=custom_color, linewidth=3, inner="box")
        
plt.title(f"Predicting HGS from anthropometric features for increasing sample sizes by gender - combined dominant and non-dominant", fontsize=10, fontweight="bold", y=1.03)

plt.xlabel("Samples", fontsize=35, fontweight="bold")
# Set xlabel, ylabel, and title
plt.xlabel("Samples")
plt.ylabel("r value", fontsize=35, fontweight="bold")
# Change x-axis tick labels
new_xticklabels = ["10%", "20%", "40%", "60%", "80%", "100%"]  # Replace with your desired labels
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=new_xticklabels, fontsize=24, weight='bold')

ymin, ymax = plt.ylim()
y_step_value = 0.02
# plt.yticks(np.arange(round(ymin/0.01)*.01, round(ymax/0.01)*.01+.03, y_step_value), fontsize=18, weight='bold', y=1.01)
plt.yticks(np.arange(0.20, 0.50, y_step_value), fontsize=18, weight='bold', y=1.01)

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='20', fontsize='18', bbox_to_anchor=(1.0005, 1), loc='upper left')
# plt.legend().remove()
# Show the plot
plt.show()
# Save the plot to a file
plt.savefig("crc_plot_by_gender.png")
# Close the figure
plt.close()

print("===== Done! =====")
embed(globals(), locals())


