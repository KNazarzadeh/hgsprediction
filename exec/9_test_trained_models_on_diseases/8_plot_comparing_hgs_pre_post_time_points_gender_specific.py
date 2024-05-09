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


from hgsprediction.load_results.load_prepared_data_for_anova import load_prepare_data_for_anova

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
anova_target = sys.argv[12]
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

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
folder_path = os.path.join("plot_hgs_comparison_pre_post_time_points_gender_specific", f"{population}", f"{target}", f"{n_samples}_matched", "pre_post_time_points")
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
###############################################################################        
xtick_labels = ['Pre time-point', 'Post time-point']

palette_male = sns.color_palette("Paired")
palette_female = sns.color_palette("PiYG")
# Map each gender-group combination to a specific color
custom_palette = {
    'male-control': palette_male[1],
    f'male-{population}': palette_male[0],
    'female-control': palette_female[0],
    f'female-{population}': palette_female[1]
}
# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(figsize=(12, 10))
sns.boxplot(data=df_both, x='gender_group_time_point', y=f"{anova_target}", hue="gender_group", palette=custom_palette, linewidth=3)

ax.legend().set_visible(False)
ax.set_xlabel(" ", fontsize=30, fontweight="bold")

# Setting the xtick labels
ax.set_xticklabels("")
# Define new tick positions and labels
new_positions = [1.5, 5.5]  # Midpoints of groups of four ticks
# Set new positions and labels
ax.set_xticks(new_positions)
ax.set_xticklabels(xtick_labels, size=25, weight='bold')

if anova_target == "hgs":
    ax.set_ylabel("Raw HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_corrected_predicted":
    ax.set_ylabel("Adjusted HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_predicted":
    ax.set_ylabel("Predicted HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_corrected_delta":
    ax.set_ylabel("Delta adjusted HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_delta":
    ax.set_ylabel("Delta HGS", fontsize=30, fontweight="bold")
    
xticks_positios_array = add_median_labels(ax)

for x_box_pos in np.arange(0,8,2):
    if x_box_pos == 0:
        idx = "pre-time_point"
        colx = "female"
    if x_box_pos == 2:
        idx = "pre-time_point"
        colx = "male"
    if x_box_pos == 4:
        idx = "post-time_point"
        colx = "female"
    if x_box_pos == 6:
        idx = "post-time_point"
        colx = "male"        
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, colx]+.5, 2, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
    ax.text((x1+x2)*.5, y+h, f"p={df_mannwhitneyu.loc[idx, colx]:.3f}", ha='center', va='bottom', fontsize=18, weight='bold',  color=col)

# if anova_target in ["hgs", "hgs_predicted", "hgs_corrected_predicted"]:
#     ax.set_yticks(range(0, 140, 20))
#     ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
#     plt.ylim(0, 140)
# elif anova_target in ["hgs_delta", "hgs_corrected_delta"]:
#     ymin = round(ax.get_ylim()[0])
#     ymax = round(ax.get_ylim()[1])
#     # ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 20))
#     ax.set_yticks(range(-60, 60, 20))
#     ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
#     plt.ylim(-60, 60)

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
folder_path = os.path.join("plot_hgs_comparison_pre_post_time_points_gender_specific", f"{population}", f"{target}", f"{n_samples}_matched", "interaction")
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
##############################################################################
# Example DataFrame preprocessing if necessary
df_interaction['gender_group'] = df_interaction['gender'] + '-' + df_interaction['group']

##############################################################################
palette_male = sns.color_palette("Paired")
palette_female = sns.color_palette("PiYG")
# Map each gender-group combination to a specific color
custom_palette = {
    'male-control': palette_male[1],
    f'male-{population}': palette_male[0],
    'female-control': palette_female[0],
    f'female-{population}': palette_female[1]
}

# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(data=df_interaction, x='gender', y=f"interaction_{anova_target}", hue='gender_group', palette=custom_palette, linewidth=3)
ax.legend().set_visible(False)
ax.set_xlabel("Interaction", fontsize=25, fontweight="bold")
ax.set_xticklabels("")

# Setting the xtick labels
if anova_target == "hgs":
    ax.set_ylabel("Raw HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_corrected_predicted":
    ax.set_ylabel("Adjusted HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_predicted":
    ax.set_ylabel("Predicted HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_corrected_delta":
    ax.set_ylabel("Delta adjusted HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_delta":
    ax.set_ylabel("Delta HGS", fontsize=30, fontweight="bold")
    
xticks_positios_array = add_median_labels(ax)

for x_box_pos in np.arange(0,4,2):
    if x_box_pos == 0:
        idx = "interaction"
        colx = "female"
    if x_box_pos == 2:
        idx = "interaction"
        colx = "male"      
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, colx]+.5, 2, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
    ax.text((x1+x2)*.5, y+h, f"p={df_mannwhitneyu.loc[idx, colx]:.3f}", ha='center', va='bottom', fontsize=18, weight='bold',  color=col)
    
    
# x_box_pos = 0
# x1 = xticks_positios_array[x_box_pos]
# x2 = xticks_positios_array[x_box_pos+1]
# y, h, col = df_yaxis_max.loc["interaction", f"{anova_target}_max_value"]+1, 2, 'k'
# ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
# ax.text((x1+x2)*.5, y+h, f"p={p_value_interaction:.3f}", ha='center', va='bottom', fontsize=18, weight='bold',  color=col)

if anova_target in ["hgs", "hgs_corrected_predicted"]:
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 10))
    ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
elif anova_target == "hgs_delta":
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 10))
    ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
elif anova_target in ["hgs_corrected_delta", "hgs_predicted"]:
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10))
    ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
file_path = os.path.join(folder_path, f"{population}_{anova_target}_interaction.png")
plt.savefig(file_path)
plt.close()


print("===== Done! End =====")
embed(globals(), locals())