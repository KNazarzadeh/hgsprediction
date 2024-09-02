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
# print("===== Done! =====")
# embed(globals(), locals())

df = data[["gender", "group", "time_point", anova_target]]

df_disorder = df[df["group"] == f"{population}"]
df_control = df[df["group"] == "control"]

df_pre_disorder = df_disorder[df_disorder['time_point']==f"pre-{population}"]
df_post_disorder = df_disorder[df_disorder['time_point']==f"post-{population}"]

df_pre_control = df_control[df_control['time_point']=="pre-control"]
df_post_control= df_control[df_control['time_point']=="post-control"]

df_female_pre_control = df_pre_control[df_pre_control['gender']==0]
df_female_post_control = df_post_control[df_post_control['gender']==0]
df_female_pre_disorder = df_pre_disorder[df_pre_disorder['gender']==0]
df_female_post_disorder = df_post_disorder[df_post_disorder['gender']==0]
df_male_pre_control = df_pre_control[df_pre_control['gender']==1]
df_male_post_control = df_post_control[df_post_control['gender']==1]
df_male_pre_disorder = df_pre_disorder[df_pre_disorder['gender']==1]
df_male_post_disorder = df_post_disorder[df_post_disorder['gender']==1]

# Assuming df_post_control and df_pre_control are defined elsewhere
df_female_interaction_control = pd.DataFrame()
df_female_interaction_disorder = pd.DataFrame()
df_male_interaction_control = pd.DataFrame()
df_male_interaction_disorder = pd.DataFrame()
# Assuming df_post_control and df_pre_control have the same indices
df_female_interaction_control[f"interaction_{anova_target}"] = df_female_post_control[f"{anova_target}"].values - df_female_pre_control[f"{anova_target}"].values
df_female_interaction_control["group"] = "control"
df_female_interaction_control["time_point"] = "Interaction"
df_female_interaction_control["gender"] = "female"

# Assuming df_post_disorder and df_pre_disorder are defined elsewhere
df_female_interaction_disorder[f"interaction_{anova_target}"] = df_female_post_disorder[f"{anova_target}"].values - df_female_pre_disorder[f"{anova_target}"].values
df_female_interaction_disorder["group"] = f"{population}"
df_female_interaction_disorder["time_point"] = "Interaction"
df_female_interaction_disorder["gender"] = "female"
# Assuming df_post_control and df_pre_control have the same indices
df_male_interaction_control[f"interaction_{anova_target}"] = df_male_post_control[f"{anova_target}"].values - df_male_pre_control[f"{anova_target}"].values
df_male_interaction_control["group"] = "control"
df_male_interaction_control["time_point"] = "Interaction"
df_male_interaction_control["gender"] = "male"

# Assuming df_post_disorder and df_pre_disorder are defined elsewhere
df_male_interaction_disorder[f"interaction_{anova_target}"] = df_male_post_disorder[f"{anova_target}"].values - df_male_pre_disorder[f"{anova_target}"].values
df_male_interaction_disorder["group"] = f"{population}"
df_male_interaction_disorder["time_point"] = "Interaction"
df_male_interaction_disorder["gender"] = "male"
# Concatenating the two DataFrames
df_interaction = pd.concat([df_female_interaction_control, df_female_interaction_disorder, df_male_interaction_control, df_male_interaction_disorder], axis=0)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df_mannwhitneyu = pd.DataFrame(index=["pre-time_point", "post-time_point", "interaction"], columns=["female", "male"])
df_yaxis_max = pd.DataFrame(index=["pre-time_point", "post-time_point", "interaction"], columns=["female", "male"])

stat_pre_female, p_value_pre_female = mannwhitneyu(df_female_pre_control[f"{anova_target}"], df_female_pre_disorder[f"{anova_target}"], nan_policy='propagate')
stat_post_female, p_value_post_female = mannwhitneyu(df_female_post_control[f"{anova_target}"], df_female_post_disorder[f"{anova_target}"], nan_policy='propagate')
stat_pre_male, p_value_pre_male = mannwhitneyu(df_male_pre_control[f"{anova_target}"], df_male_pre_disorder[f"{anova_target}"], nan_policy='propagate')
stat_post_male, p_value_post_male = mannwhitneyu(df_male_post_control[f"{anova_target}"], df_male_post_disorder[f"{anova_target}"], nan_policy='propagate')

df_mannwhitneyu.loc["pre-time_point", "female"] = p_value_pre_female
df_mannwhitneyu.loc["post-time_point", "female"] = p_value_post_female
df_mannwhitneyu.loc["pre-time_point", "male"] = p_value_pre_male
df_mannwhitneyu.loc["post-time_point", "male"] = p_value_post_male

stat_interaction_female, p_value_interaction_female = stats.mannwhitneyu(df_female_interaction_control[f"interaction_{anova_target}"], df_female_interaction_disorder[f"interaction_{anova_target}"])
stat_interaction_male, p_value_interaction_male = stats.mannwhitneyu(df_male_interaction_control[f"interaction_{anova_target}"], df_male_interaction_disorder[f"interaction_{anova_target}"])

df_mannwhitneyu.loc["interaction", "female"] = p_value_interaction_female
df_mannwhitneyu.loc["interaction", "male"] = p_value_interaction_male

max_value_pre_female = max(df_female_pre_control[f"{anova_target}"].max(), df_female_pre_disorder[f"{anova_target}"].max())
max_value_post_female = max(df_female_post_control[f"{anova_target}"].max(), df_female_post_disorder[f"{anova_target}"].max())
max_value_pre_male = max(df_male_pre_control[f"{anova_target}"].max(), df_male_pre_disorder[f"{anova_target}"].max())
max_value_post_male = max(df_male_post_control[f"{anova_target}"].max(), df_male_post_disorder[f"{anova_target}"].max())

max_value_interaction_female = max(df_female_interaction_control[f"interaction_{anova_target}"].max(), df_female_interaction_disorder[f"interaction_{anova_target}"].max())
max_value_interaction_male = max(df_male_interaction_control[f"interaction_{anova_target}"].max(), df_male_interaction_disorder[f"interaction_{anova_target}"].max())


df_yaxis_max.loc["pre-time_point", "female"] = max_value_pre_female
df_yaxis_max.loc["post-time_point", "female"] = max_value_post_female
df_yaxis_max.loc["pre-time_point", "male"] = max_value_pre_male
df_yaxis_max.loc["post-time_point", "male"] = max_value_post_male

df_yaxis_max.loc["interaction", "female"] = max_value_interaction_female
df_yaxis_max.loc["interaction", "male"] = max_value_interaction_male

###############################################################################
def add_median_labels(ax, fmt='.3f'):
    xticks_positios_array = []
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        # text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=18)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        # text.set_path_effects([
        #     path_effects.Stroke(linewidth=4, foreground=median.get_color()),
        #     path_effects.Normal(),
        # ])
        xticks_positios_array.append(x)
    return xticks_positios_array

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Replace values based on time_points
df_pre = pd.concat([df_female_pre_control, df_female_pre_disorder, df_male_pre_control, df_male_pre_disorder], axis=0)
df_post = pd.concat([df_female_post_control, df_female_post_disorder, df_male_post_control, df_male_post_disorder], axis=0)

df_pre.loc[df_pre['gender'] == 0, 'gender'] = 'female'
df_pre.loc[df_pre['gender'] == 1, 'gender'] = 'male'
df_post.loc[df_post['gender'] == 0, 'gender'] = 'female'
df_post.loc[df_post['gender'] == 1, 'gender'] = 'male'

df_pre.loc[df_pre['time_point'].str.contains('pre-'), 'time_point'] = 'Pre-time_point'
df_post.loc[df_post['time_point'].str.contains('post-'), 'time_point'] = 'Post-time_point'

df_both = pd.concat([df_pre, df_post])
df_both.loc[:, 'gender_group_time_point'] = df_both.loc[:, 'gender'] + "-" + df_both.loc[:,'group'] + "-" + df_both.loc[:, 'time_point']

# Example DataFrame preprocessing if necessary
df_both['gender_group'] = df_both['gender'] + '-' + df_both['group']

df_female = df_both[df_both['gender']=='female']
df_male = df_both[df_both['gender']=='male']

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
folder_path = os.path.join("plot_nonparametric_gender_specific_side_by_side_plots", f"{population}", f"{target}", f"{n_samples}_matched", "pre_post_time_points")
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
###############################################################################        
xtick_labels = ['Pre time-point', 'Post time-point']

palette_pastel = sns.color_palette("pastel")
palette_deep = sns.color_palette("deep")
palette_dark = sns.color_palette("dark")

# Map each gender-group combination to a specific color
custom_palette = {
    'male-control-Pre-time_point': palette_pastel[2],
    'male-control-Post-time_point': palette_deep[2],
    'male-depression-Pre-time_point': palette_pastel[0],
    'male-depression-Post-time_point': palette_deep[0],
    'female-control-Pre-time_point': palette_pastel[2],
    'female-control-Post-time_point': palette_deep[2],
    'female-depression-Pre-time_point': palette_pastel[6],
    'female-depression-Post-time_point': palette_dark[6],
}
###############################################################################
# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(1,2, figsize=(10, 5))
sns.boxplot(data=df_male, x='gender_group_time_point', y=f"{anova_target}", hue="gender_group_time_point", palette=custom_palette, ax=ax[0], width=2)
sns.boxplot(data=df_female, x='gender_group_time_point', y=f"{anova_target}", hue="gender_group_time_point", palette=custom_palette,ax=ax[1], width=2)
#-----------------------------------------------------------#
# Remove legends from individual plots
ax[0].legend().set_visible(False)
ax[1].legend().set_visible(False)
#-----------------------------------------------------------#
# Set x-labels for both plots
ax[0].set_xlabel(" ")
ax[1].set_xlabel(" ")
#-----------------------------------------------------------#
# Setting the xtick labels
ax[0].set_xticklabels("")
ax[1].set_xticklabels("")
#-----------------------------------------------------------#
# Define new tick positions and labels
new_positions = [0.1, 2.9]  # Midpoints of groups of four ticks
# Set new positions and labels
for axes in [ax[0], ax[1]]:
    axes.set_xticks(new_positions)  # Set tick positions
    axes.set_xticklabels(xtick_labels, fontsize=16)  # Set tick labels
#-----------------------------------------------------------#
if anova_target == "hgs":
    ax[0].set_ylabel("True HGS", fontsize=16)
    ax[1].set_ylabel("True HGS", fontsize=16)
elif anova_target == "hgs_corrected_predicted":
    ax[0].set_ylabel(r"$\mathrm{HGS}^c$", fontsize=16)
    ax[1].set_ylabel(r"$\mathrm{HGS}^c$", fontsize=16)
elif anova_target == "hgs_predicted":
    ax[0].set_ylabel("Predicted HGS", fontsize=16)
    ax[1].set_ylabel("Predicted HGS", fontsize=16)
    
elif anova_target == "hgs_corrected_delta":
    ax[0].set_ylabel(r"$\Delta \mathrm{HGS}^c$", fontsize=16)
    ax[1].set_ylabel(r"$\Delta \mathrm{HGS}^c$", fontsize=16)
    
elif anova_target == "hgs_delta":
    ax[0].set_ylabel(r"$\Delta {HGS}$", fontsize=16)
    ax[1].set_ylabel(r"$\Delta {HGS}$", fontsize=16)
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
xticks_positios_array = add_median_labels(ax[0])
xticks_positios_array = add_median_labels(ax[1])

for x_box_pos in np.arange(0,3,2):
    if x_box_pos == 0:
        idx = "pre-time_point"
    if x_box_pos == 2:
        idx = "post-time_point"    
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, 'male']+2, 1, 'k'
    ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
    ax[0].text((x1+x2)*.5, y+h, f"$p$={df_mannwhitneyu.loc[idx, 'male']:.3f}", ha='center', va='bottom', fontsize=10, color=col)
    
    y, h, col = df_yaxis_max.loc[idx, 'female']+2, 1, 'k'
    ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
    ax[1].text((x1+x2)*.5, y+h, f"$p$={df_mannwhitneyu.loc[idx, 'female']:.3f}", ha='center', va='bottom', fontsize=10, color=col)
#-----------------------------------------------------------#
# Format y-tick labels to one decimal place if they are round
# Get x and y limits for the first subplot first row
ymin0, ymax0 = ax[0].get_ylim()
# Get x and y limits for the second subplot first row
ymin1, ymax1 = ax[1].get_ylim()
# Find the common y-axis limits
ymin = min(ymin0, ymin1)
ymax = max(ymax0, ymax1)
ystep_value = 5

yticks_range = np.arange(math.floor(ymin / 10) * 10, math.ceil(ymax / 10) * 10+ystep_value, ystep_value)
#-----------------------------------------------------------#
# Set the y-ticks for both subplots
ax[0].set_yticks(yticks_range)
ax[1].set_yticks(yticks_range)

#-----------------------------------------------------------#
# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
file_path = os.path.join(folder_path, f"{population}_{anova_target}.png")
plt.savefig(file_path)
plt.close()

# print("===== Done! =====")
# embed(globals(), locals())

##############################################################################
folder_path = os.path.join("plot_nonparametric_gender_specific_side_by_side_plots", f"{population}", f"{target}", f"{n_samples}_matched", "interaction")
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
##############################################################################
# Example DataFrame preprocessing if necessary
df_interaction['gender_group'] = df_interaction['gender'] + '-' + df_interaction['group']

##############################################################################
palette_Paired = sns.color_palette("Paired")
palette_hls =sns.color_palette("hls", 8)
# Map each gender-group combination to a specific color
custom_palette = {
    'male-control': palette_Paired[3],
    f'male-{population}': palette_Paired[1],
    'female-control': palette_Paired[3],
    f'female-{population}': palette_hls[7]
}
##############################################################################
xtick_labels = ["Females", "Males"]
# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df_interaction, x='gender', y=f"interaction_{anova_target}", hue='gender_group', palette=custom_palette, width=2)
#-----------------------------------------------------------#
# Remove legends from individual plots
ax.legend().set_visible(False)
#-----------------------------------------------------------#
# Setting the xlabel
ax.set_xlabel("Interaction", fontsize=16)
#-----------------------------------------------------------#
# Define new tick positions and labels
new_positions = [0, 1]  # Midpoints of groups of four ticks
# Set new positions and labels
ax.set_xticks(new_positions)  # Set tick positions
ax.set_xticklabels(xtick_labels, fontsize=16)  # Set tick labels
#-----------------------------------------------------------#
if anova_target == "hgs":
    ax.set_ylabel("True HGS", fontsize=16)
elif anova_target == "hgs_corrected_predicted":
    ax.set_ylabel(r"$\mathrm{HGS}^c$", fontsize=16)
elif anova_target == "hgs_predicted":
    ax.set_ylabel("Predicted HGS", fontsize=16)
    
elif anova_target == "hgs_corrected_delta":
    ax.set_ylabel(r"$\Delta \mathrm{HGS}^c$", fontsize=16)
    
elif anova_target == "hgs_delta":
    ax.set_ylabel(r"$\Delta {HGS}$", fontsize=16)
#-----------------------------------------------------------#
xticks_positios_array = add_median_labels(ax)
#-----------------------------------------------------------#
for x_box_pos in np.arange(0,3,2):
    if x_box_pos == 0:
        idx = "interaction"
        colx = "female"
    if x_box_pos == 2:
        idx = "interaction"
        colx = "male"      
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, colx]+1, .5, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
    ax.text((x1+x2)*.5, y+h, f"$p$={df_mannwhitneyu.loc[idx, colx]:.3f}", ha='center', va='bottom', fontsize=10,  color=col)
    
#-----------------------------------------------------------#
# if anova_target in ["hgs", "hgs_corrected_predicted"]:
#     ymin = round(ax.get_ylim()[0])
#     ymax = round(ax.get_ylim()[1])
#     ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 10))
#     ax.set_yticklabels(ax.get_yticks())
# elif anova_target == "hgs_delta":
#     ymin = round(ax.get_ylim()[0])
#     ymax = round(ax.get_ylim()[1])
#     ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 10))
#     ax.set_yticklabels(ax.get_yticks())
# elif anova_target in ["hgs_corrected_delta", "hgs_predicted"]:
#     ymin = round(ax.get_ylim()[0])
#     ymax = round(ax.get_ylim()[1])
#     ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10))
#     ax.set_yticklabels(ax.get_yticks())
    
#-----------------------------------------------------------#
# Iterate over each subplot to change the font size for tick labels
ax.tick_params(axis='both', labelsize=12, direction='out', length=5)
#-----------------------------------------------------------#
# Set the color of the plot's spines to black for both subplots
for spine in ax.spines.values():
    spine.set_color('black')
#-----------------------------------------------------------#
# Set the y-ticks for both subplots
ax.set_yticks(yticks_range)
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