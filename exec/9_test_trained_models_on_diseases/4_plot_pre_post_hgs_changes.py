import math
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

from hgsprediction.load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results
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
boxplot_target = sys.argv[12]
first_event = sys.argv[13]
##############################################################################
folder_path = os.path.join("plot_paired", f"{population}", f"{first_event}", f"{target}", f"{n_samples}_matched")
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

##############################################################################
df_disorder_matched_female, df_mathced_controls_female = load_disorder_matched_samples_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "female",
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
    first_event,
)

# Replace 0 with "Male" in the 'Gender' column
df_disorder_matched_female['gender'] = df_disorder_matched_female['gender'].replace(0, 'Female')
df_mathced_controls_female['gender'] = df_mathced_controls_female['gender'].replace(0, 'Female')

df_disorder_matched_male, df_mathced_controls_male = load_disorder_matched_samples_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "male",
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
    first_event,
)
# print("===== Done! End =====")
# embed(globals(), locals())
##############################################################################
# Replace 0 with "Male" in the 'Gender' column
df_disorder_matched_male['gender'] = df_disorder_matched_male['gender'].replace(1, 'Male')
df_mathced_controls_male['gender'] = df_mathced_controls_male['gender'].replace(1, 'Male')
##############################################################################
if boxplot_target == "hgs":
    y_label = "Raw HGS"
    # Example data for pre-hgs and post-hgs for female and male
    disorder_pre_female = df_disorder_matched_female[[f'1st_pre-{population}_{target}', 'gender']]
    disorder_pre_male = df_disorder_matched_male[[f'1st_pre-{population}_{target}', 'gender']]
    disorder_post_female = df_disorder_matched_female[[f'1st_post-{population}_{target}', 'gender']]
    disorder_post_male = df_disorder_matched_male[[f'1st_post-{population}_{target}', 'gender']]
    # Example data for pre-hgs and post-hgs for female and male
    control_pre_female = df_mathced_controls_female[[f'1st_pre-{population}_{target}', 'gender']]
    control_pre_male = df_mathced_controls_male[[f'1st_pre-{population}_{target}', 'gender']]
    control_post_female = df_mathced_controls_female[[f'1st_post-{population}_{target}', 'gender']]
    control_post_male = df_mathced_controls_male[[f'1st_post-{population}_{target}', 'gender']]

elif boxplot_target in ["predicted", "corrected_predicted"]:
    y_label = "Predicted HGS"
    # Example data for pre-hgs and post-hgs for female and male
    disorder_pre_female = df_disorder_matched_female[[f'1st_pre-{population}_{target}_{boxplot_target}', 'gender']]
    disorder_pre_male = df_disorder_matched_male[[f'1st_pre-{population}_{target}_{boxplot_target}', 'gender']]
    disorder_post_female = df_disorder_matched_female[[f'1st_post-{population}_{target}_{boxplot_target}', 'gender']]
    disorder_post_male = df_disorder_matched_male[[f'1st_post-{population}_{target}_{boxplot_target}', 'gender']]
    # Example data for pre-hgs and post-hgs for female and male
    control_pre_female = df_mathced_controls_female[[f'1st_pre-{population}_{target}_{boxplot_target}', 'gender']]
    control_pre_male = df_mathced_controls_male[[f'1st_pre-{population}_{target}_{boxplot_target}', 'gender']]
    control_post_female = df_mathced_controls_female[[f'1st_post-{population}_{target}_{boxplot_target}', 'gender']]
    control_post_male = df_mathced_controls_male[[f'1st_post-{population}_{target}_{boxplot_target}', 'gender']]
       
elif boxplot_target in ["delta", "corrected_delta"]:
    # Example data for pre-hgs and post-hgs for female and male
    disorder_pre_female = df_disorder_matched_female[[f'1st_pre-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
    disorder_pre_male = df_disorder_matched_male[[f'1st_pre-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
    disorder_post_female = df_disorder_matched_female[[f'1st_post-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
    disorder_post_male = df_disorder_matched_male[[f'1st_post-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
    # Example data for pre-hgs and post-hgs for female and male
    control_pre_female = df_mathced_controls_female[[f'1st_pre-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
    control_pre_male = df_mathced_controls_male[[f'1st_pre-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
    control_post_female = df_mathced_controls_female[[f'1st_post-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
    control_post_male = df_mathced_controls_male[[f'1st_post-{population}_{target}_{boxplot_target}(true-predicted)', 'gender']]
# print("===== Done! End =====")
# embed(globals(), locals())
# Combine data into a DataFrame
data_disorder_pre = pd.concat([disorder_pre_female, disorder_pre_male])
data_disorder_post = pd.concat([disorder_post_female, disorder_post_male])

# Combine data into a DataFrame
data_control_pre = pd.concat([control_pre_female, control_pre_male])
data_control_post = pd.concat([control_post_female, control_post_male])


# Rename columns for clarity
data_disorder_pre.columns = [f'{boxplot_target}', 'Gender']
data_disorder_post.columns = [f'{boxplot_target}', 'Gender']
# Rename columns for clarity
data_control_pre.columns = [f'{boxplot_target}', 'Gender']
data_control_post.columns = [f'{boxplot_target}', 'Gender']


# Add a new column for Time_point
data_disorder_pre['Time_point'] = 'Pre-Time_point'
data_disorder_post['Time_point'] = 'Post-Time_point'
# Add a new column for Time_point
data_control_pre['Time_point'] = 'Pre-Time_point'
data_control_post['Time_point'] = 'Post-Time_point'

# Check if the indices are in the same order
if data_control_pre.index.equals(data_control_post.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")
    

# Check if the indices are in the same order
if data_disorder_pre.index.equals(data_disorder_post.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")
  
# Combine pre and post data
data_disorder = pd.concat([data_disorder_pre, data_disorder_post])

data_control = pd.concat([data_control_pre, data_control_post])
# print("===== Done! End =====")
# embed(globals(), locals())
###############################################################################
if boxplot_target == "hgs":
    y_label = "Raw HGS"
elif boxplot_target == "predicted":
    y_label = "Predicted HGS"
elif boxplot_target == "corrected_predicted":
    y_label = "Adjusted predicted HGS"
elif boxplot_target == "delta":
    y_label = "Delta HGS"
elif boxplot_target == "corrected_delta":
    y_label = "Delta adjusted HGS"
###############################################################################
custom_palette = {'Pre-Time_point':'lightgrey', 'Post-Time_point':'grey'}
# Set the style of seaborn
sns.set_style("whitegrid")

# Create the boxplot
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.boxplot(x='Time_point', y=f'{boxplot_target}', data=data_disorder, palette=custom_palette, ax=ax[0])
sns.boxplot(x='Time_point', y=f'{boxplot_target}', data=data_control, palette=custom_palette, ax=ax[1])

# Adding connecting lines and scatter points with gender-based colors for disorder data
# for idx in data_disorder_pre.index:
for subject_id, row in data_disorder_pre.iterrows():
    subject_gender = data_disorder_pre.loc[subject_id, 'Gender']
    pre_hgs = data_disorder_pre.loc[subject_id, f'{boxplot_target}']
    post_hgs = data_disorder_post.loc[subject_id, f'{boxplot_target}']  # Extract scalar value from series
    # Plot connecting lines and scatter points
    line_color = 'purple' if subject_gender == "Female" else 'blue'
    marker_color = 'purple' if subject_gender == "Female" else 'blue'
    ax[0].plot([0, 1], [pre_hgs, post_hgs], color=line_color, linewidth=1, alpha=.5)
    ax[0].scatter([0, 1], [pre_hgs, post_hgs], color=marker_color, marker='o', s=30)

# Adding connecting lines and scatter points with gender-based colors for control data
# for idx in data_control_pre.index:
for subject_id, row in data_control_pre.iterrows():
    subject_gender = row['Gender']
    pre_hgs = row[boxplot_target]
    post_hgs = data_control_post[data_control_post.index==subject_id][boxplot_target].iloc[0] # Extract scalar value from series
    # Plot connecting lines and scatter points
    line_color = 'purple' if subject_gender == "Female" else 'blue'
    marker_color = 'purple' if subject_gender == "Female" else 'blue'
    ax[1].plot([0, 1], [pre_hgs, post_hgs], color=line_color, linewidth=1, alpha=.5)
    ax[1].scatter([0, 1], [pre_hgs, post_hgs], color=marker_color, marker='o', s=30)

# Set y-axis limits and ticks to be the same for both subplots
y_min = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
y_max = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])

# Round y-axis minimum value to 1 integer lower
y_min = math.floor(y_min/10)*10
# Round y-axis maximum value to 1 integer higher
y_max = math.ceil(y_max/10)*10

for axis in ax:
    if boxplot_target in ["hgs", "predicted", "corrected_predicted"]:
        axis.set_ylim(y_min, y_max)
        axis.set_yticks(range(y_min, y_max, 10))
    else:
        axis.set_ylim(y_min, y_max)
        axis.set_yticks(range(y_min, y_max, 5))
# Set titles for subplots
ax[0].set_title(f'{population.capitalize()}(N={len(data_disorder_pre)}) - Female(N={len(disorder_pre_female)}), Male(N={len(disorder_pre_male)})', fontsize=16, fontweight='bold')
ax[1].set_title(f'Matched Controls(N={len(data_control_pre)}) - Female(N={len(control_pre_female)}), Male(N={len(control_pre_male)})', fontsize=16, fontweight='bold')

# Increase the size of x-axis tick labels for each subplot
for axis in ax:
    axis.tick_params(axis='both', which='major', labelsize=16)

# Labeling the plot with bold and larger fonts for each subplot
for axis in ax:
    axis.set_xlabel('')
    axis.set_ylabel(f'{y_label}', fontsize=16, fontweight='bold')

plt.show()
file_path = os.path.join(folder_path, f"box_plot_pair_plot_{population}_{boxplot_target}.png")
plt.savefig(file_path)
plt.close()
print("===== Done! End =====")
embed(globals(), locals())

