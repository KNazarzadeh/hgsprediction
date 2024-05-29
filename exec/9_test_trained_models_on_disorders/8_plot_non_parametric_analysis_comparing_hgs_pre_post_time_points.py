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
anova_target = sys.argv[12]
first_event = sys.argv[13]
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
    first_event,
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

# Assuming df_post_control and df_pre_control are defined elsewhere
df_interaction_control = pd.DataFrame()
df_interaction_disorder = pd.DataFrame()

# Assuming df_post_control and df_pre_control have the same indices
df_interaction_control[f"interaction_{anova_target}"] = df_post_control[f"{anova_target}"].values - df_pre_control[f"{anova_target}"].values
df_interaction_control["group"] = "control"
df_interaction_control["time_point"] = "Interaction"

# Assuming df_post_disorder and df_pre_disorder are defined elsewhere
df_interaction_disorder[f"interaction_{anova_target}"] = df_post_disorder[f"{anova_target}"].values - df_pre_disorder[f"{anova_target}"].values
df_interaction_disorder["group"] = f"{population}"
df_interaction_disorder["time_point"] = "Interaction"

# Concatenating the two DataFrames
df_interaction = pd.concat([df_interaction_control, df_interaction_disorder], axis=0)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df_mannwhitneyu = pd.DataFrame(index=["pre-time_point", "post-time_point", "interaction"])
df_yaxis_max = pd.DataFrame(index=["pre-time_point", "post-time_point", "interaction"])

stat_pre, p_value_pre = mannwhitneyu(df_pre_control[f"{anova_target}"], df_pre_disorder[f"{anova_target}"], nan_policy='propagate')
stat_post, p_value_post = mannwhitneyu(df_post_control[f"{anova_target}"], df_post_disorder[f"{anova_target}"], nan_policy='propagate')

df_mannwhitneyu.loc["pre-time_point", f"{anova_target}_p_value"] = p_value_pre
df_mannwhitneyu.loc["post-time_point", f"{anova_target}_p_value"] = p_value_post

stat_interaction, p_value_interaction = stats.mannwhitneyu(df_interaction_control[f"interaction_{anova_target}"], df_interaction_disorder[f"interaction_{anova_target}"])

df_mannwhitneyu.loc["interaction", f"{anova_target}_p_value"] = p_value_interaction

max_value_pre = max(df_pre_control[f"{anova_target}"].max(), df_pre_disorder[f"{anova_target}"].max())
max_value_post = max(df_post_control[f"{anova_target}"].max(), df_post_disorder[f"{anova_target}"].max())
max_value_interaction = max(df_interaction_control[f"interaction_{anova_target}"].max(), df_interaction_disorder[f"interaction_{anova_target}"].max())


df_yaxis_max.loc["pre-time_point", f"{anova_target}_max_value"] = max_value_pre
df_yaxis_max.loc["post-time_point", f"{anova_target}_max_value"] = max_value_post
df_yaxis_max.loc["interaction", f"{anova_target}_max_value"] = max_value_interaction
# print("===== Done! =====")
# embed(globals(), locals())
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
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=18)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=4, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array

###############################################################################
# Replace values based on time_points
df.loc[data['time_point'].str.contains('pre-'), 'time_point'] = 'Pre-time_point'
df.loc[data['time_point'].str.contains('post-'), 'time_point'] = 'Post-time_point'
###############################################################################
folder_path = os.path.join("plot_non_parametric_analysis", f"{population}", f"{first_event}", f"{target}", f"{n_samples}_matched", "pre_post_time_points")

if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
###############################################################################        
xtick_labels = ['Pre-time_point', 'Post-time_point']

palette_control = sns.color_palette("Paired")
palette_disorder = sns.color_palette("PiYG")
custome_palette = [palette_control[1], palette_disorder[0]]

# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(data=df, x='time_point', y=f"{anova_target}", hue='group', palette=custome_palette, linewidth=3)
ax.legend().set_visible(False)
ax.set_xlabel(" ", fontsize=30, fontweight="bold")

# Setting the xtick labels
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

for x_box_pos in np.arange(0,4,2):
    if x_box_pos == 0:
        idx = "pre-time_point"
    if x_box_pos == 2:
        idx = "post-time_point"
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, f"{anova_target}_max_value"]+.5, 2, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
    ax.text((x1+x2)*.5, y+h, f"p={df_mannwhitneyu.loc[idx, f'{anova_target}_p_value']:.3f}", ha='center', va='bottom', fontsize=18, weight='bold',  color=col)

if anova_target in ["hgs", "hgs_predicted", "hgs_corrected_predicted"]:
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    # ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 20))
    ax.set_yticks(range(ymin, ymax, 20))
    ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
    plt.ylim(ymin, ymax)
    
elif anova_target in ["hgs_delta", "hgs_corrected_delta"]:
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    # ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 20))
    ax.set_yticks(range(ymin, ymax, 5))
    ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
    plt.ylim(ymin, ymax)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
file_path = os.path.join(folder_path, f"{population}_{anova_target}.png")
plt.savefig(file_path)
plt.close()


##############################################################################
folder_path = os.path.join("plot_non_parametric_analysis", f"{population}", f"{first_event}", f"{target}", f"{n_samples}_matched", "interaction")

if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
##############################################################################
# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(data=df_interaction, x='time_point', y=f"interaction_{anova_target}", hue='group', palette=custome_palette, linewidth=3)
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

x_box_pos = 0
x1 = xticks_positios_array[x_box_pos]
x2 = xticks_positios_array[x_box_pos+1]
y, h, col = df_yaxis_max.loc["interaction", f"{anova_target}_max_value"]+1, 2, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
ax.text((x1+x2)*.5, y+h, f"p={p_value_interaction:.3f}", ha='center', va='bottom', fontsize=18, weight='bold',  color=col)

if anova_target in ["hgs", "hgs_corrected_predicted"]:
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 5))
    ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
elif anova_target == "hgs_delta":
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 5))
    ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
elif anova_target in ["hgs_corrected_delta", "hgs_predicted"]:
    ymin = round(ax.get_ylim()[0])
    ymax = round(ax.get_ylim()[1])
    ax.set_yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 5))
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