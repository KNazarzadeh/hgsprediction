import sys
import os
import numpy as np
import pandas as pd
import math
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
from scipy import stats


from hgsprediction.load_results.anova.load_prepared_data_for_anova import load_prepare_data_for_anova

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
confound_status = sys.argv[5]
n_repeats = sys.argv[6]
n_folds = sys.argv[7]
disorder_cohort = sys.argv[8]
visit_session = sys.argv[9]
n_samples = sys.argv[10]
target = sys.argv[11]
firts_event = sys.argv[12]
anova_target = sys.argv[13]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

##############################################################################
# Load data for ANOVA
data = load_prepare_data_for_anova(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
    firts_event,
)

df = data[["gender", "group", "time_point", anova_target]]

data_female = df[df['gender'] == 0]
data_male = df[df['gender'] == 1]

df_female_pre_control = data_female[data_female['time_point']=="pre-control"]
df_female_post_control = data_female[data_female['time_point']=="post-control"]
df_female_pre_control = df_female_pre_control.reindex(df_female_post_control.index)

df_female_pre_disorder = data_female[data_female['time_point']==f"pre-{population}"]
df_female_post_disorder = data_female[data_female['time_point']==f"post-{population}"]
df_female_pre_disorder = df_female_pre_disorder.reindex(df_female_post_disorder.index)

df_male_pre_control = data_male[data_male['time_point']=="pre-control"]
df_male_post_control = data_male[data_male['time_point']=="post-control"]
df_male_pre_control = df_male_pre_control.reindex(df_male_post_control.index)

df_male_pre_disorder = data_male[data_male['time_point']==f"pre-{population}"]
df_male_post_disorder = data_male[data_male['time_point']==f"post-{population}"]
df_male_pre_disorder = df_male_pre_disorder.reindex(df_male_post_disorder.index)

###############################################################################
folder_path = os.path.join("plot_posthoc_gender_specific_side_by_side_plots", f"{population}", f"{target}", f"{n_samples}_matched", "interaction")
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
##############################################################################
# Calculate mean values for each condition and time point
female_mean_values = {
    "control_pre-time_point": df_female_pre_control[f'{anova_target}'].mean(),
    "control_post-time_point": df_female_post_control[f'{anova_target}'].mean(),
    f"{population}_pre-time_point": df_female_pre_disorder[f'{anova_target}'].mean(),
    f"{population}_post-time_point": df_female_post_disorder[f'{anova_target}'].mean()
}

# Create the DataFrame directly from the dictionary and split the time_point column
df_female_posthoc = pd.DataFrame(list(female_mean_values.items()), columns=["time_point", "mean_value"])
df_female_posthoc[['group', 'time']] = df_female_posthoc['time_point'].str.split('_', n=1, expand=True)

# Calculate mean values for each condition and time point
male_mean_values = {
    "control_pre-time_point": df_male_pre_control[f'{anova_target}'].mean(),
    "control_post-time_point": df_male_post_control[f'{anova_target}'].mean(),
    f"{population}_pre-time_point": df_male_pre_disorder[f'{anova_target}'].mean(),
    f"{population}_post-time_point": df_male_post_disorder[f'{anova_target}'].mean()
}

# Create the DataFrame directly from the dictionary and split the time_point column
df_male_posthoc = pd.DataFrame(list(male_mean_values.items()), columns=["time_point", "mean_value"])
df_male_posthoc[['group', 'time']] = df_male_posthoc['time_point'].str.split('_', n=1, expand=True)

##############################################################################
# palette_Paired = sns.color_palette("Paired")
# palette_hls =sns.color_palette("hls", 8)
# Map each gender-group combination to a specific color
custom_palette_male = {
    'control': 'black',
    f'{population}': 'black',
}
custom_palette_female = {
    'control': 'black',
    f'{population}': 'black',
}

custom_markers = {
    'control': '^',
    f'{population}': 'o',
}

custom_linestyle = {
    'control': '--',
    f'{population}': '-',
}
# print("===== Done! End =====")
# embed(globals(), locals())
##############################################################################
xtick_labels = ['Pre time-point', 'Post time-point']
# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(1,2, figsize=(10, 5))
sns.pointplot(data=df_male_posthoc, x='time', y='mean_value', hue='group', 
              markers=[custom_markers[g] for g in df_male_posthoc['group'].unique()], 
              linestyles=[custom_linestyle[g] for g in df_male_posthoc['group'].unique()], 
              palette=custom_palette_male, 
              ax=ax[0])
sns.pointplot(data=df_female_posthoc, x='time', y='mean_value', hue='group', 
              markers=[custom_markers[g] for g in df_male_posthoc['group'].unique()], 
              linestyles=[custom_linestyle[g] for g in df_male_posthoc['group'].unique()], 
              palette=custom_palette_female, 
              ax=ax[1])

#-----------------------------------------------------------#
# Remove legends from individual plots
ax[0].legend().set_visible(False)
ax[1].legend().set_visible(False)
# Add legend outside the second plot (ax[1] only)
# ax[1].legend(
#     title="Group",
#     bbox_to_anchor=(1.02, 1),  # Adjust these values to position the legend outside ax[1]
#     loc='upper left',
#     borderaxespad=0.)
#-----------------------------------------------------------#
# Set x-labels for both plots
ax[0].set_xlabel(" ")
ax[1].set_xlabel(" ")
#-----------------------------------------------------------#
# Setting the xtick labels
ax[0].set_xticklabels("")
ax[1].set_xticklabels("")
#-----------------------------------------------------------#
ax[0].set_xticklabels(xtick_labels, fontsize=14)  # Set tick labels
ax[1].set_xticklabels(xtick_labels, fontsize=14)  # Set tick labels
#-----------------------------------------------------------#
ax[0].set_ylabel("Mean values", fontsize=16)
ax[1].set_ylabel("Mean values", fontsize=16)
#-----------------------------------------------------------#
# Iterate over each subplot to change the font size for tick labels
for axis in ax.flatten():
    axis.tick_params(axis='both', labelsize=12, direction='out', length=5)
#-----------------------------------------------------------#
# Axes titles
ax[0].set_title("Males", fontsize=12, fontweight="bold")            
ax[1].set_title("Females", fontsize=12, fontweight="bold")
#-----------------------------------------------------------#
# Set the color of the plot's spines to black for both subplots
for ax_subplot in ax:
    for spine in ax_subplot.spines.values():
        spine.set_color('black')

#-----------------------------------------------------------#
# Format y-tick labels to one decimal place if they are round
# Get x and y limits for the first subplot first row
ymin0, ymax0 = ax[0].get_ylim()
# Get x and y limits for the second subplot first row
ymin1, ymax1 = ax[1].get_ylim()
# Find the common y-axis limits
ymin = min(ymin0, ymin1)
ymax = max(ymax0, ymax1)
#-----------------------------------------------------------#
if anova_target == "true_hgs":
    ystep_value = 5
    yticks_range = np.arange(math.floor(ymin / 10) * 10, math.ceil(ymax / 10) * 10+5, ystep_value)
    
elif anova_target == "hgs_corrected_delta":
    ystep_value = 2.5
    yticks_range = np.arange(-5,  math.ceil(ymax / 10) * 10+2.5, ystep_value)
#-----------------------------------------------------------#
# Set the y-ticks for both subplots
ax[0].set_yticks(yticks_range)
ax[1].set_yticks(yticks_range)

#-----------------------------------------------------------#
# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
file_path = os.path.join(folder_path, f"{population}_{anova_target}_interaction.png")
plt.savefig(file_path)
plt.close()


print("===== Done! End =====")
embed(globals(), locals())