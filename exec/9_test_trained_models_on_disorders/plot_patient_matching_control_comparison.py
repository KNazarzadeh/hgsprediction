import math
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from hgsprediction.load_results.anova.load_prepared_data_for_anova import load_prepare_data_for_anova

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

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
plot_target = sys.argv[12]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
    
df = load_prepare_data_for_anova(
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

df.loc[df["disorder_episode"].str.contains("pre"), "episode"] = "Pre-condition"
df.loc[df["disorder_episode"].str.contains("post"), "episode"] = "Post-condition"


df_disorder = df[df["treatment"] == f"{population}"]
df_control = df[df["treatment"] == "control"]

df_disorder_pre = df_disorder[df_disorder['disorder_episode']==f"pre-{population}"]
df_disorder_post = df_disorder[df_disorder['disorder_episode']==f"post-{population}"]

df_control_pre = df_control[df_control['disorder_episode']=="pre-control"]
df_control_post = df_control[df_control['disorder_episode']=="post-control"]


###############################################################################
df_yaxis_max = pd.DataFrame()
ttest = pd.DataFrame()

stat_pre, p_value_pre = ttest_ind(df_control_pre[plot_target], df_disorder_pre[plot_target])
stat_post, p_value_post = ttest_ind(df_control_post[plot_target], df_disorder_post[plot_target])

# female
stat_pre, p_value_female_pre = ttest_ind(df_control_pre[df_control_pre['gender']==0][plot_target], df_disorder_pre[df_disorder_pre['gender']==0][plot_target])
stat_post, p_value_female_post = ttest_ind(df_control_post[df_control_post['gender']==0][plot_target], df_disorder_post[df_disorder_post['gender']==0][plot_target])

# male
stat_pre, p_value_male_pre = ttest_ind(df_control_pre[df_control_pre['gender']==1][plot_target], df_disorder_pre[df_disorder_pre['gender']==1][plot_target])
stat_post, p_value_male_post = ttest_ind(df_control_post[df_control_post['gender']==1][plot_target], df_disorder_post[df_disorder_post['gender']==1][plot_target])

ttest.loc["pre-episode", f"{plot_target}_p_value"] = p_value_pre
ttest.loc["post-episode", f"{plot_target}_p_value"] = p_value_post

ttest.loc["pre-episode_female", f"{plot_target}_p_value"] = p_value_female_pre
ttest.loc["post-episode_female", f"{plot_target}_p_value"] = p_value_female_post

ttest.loc["pre-episode_male", f"{plot_target}_p_value"] = p_value_male_pre
ttest.loc["post-episode_male", f"{plot_target}_p_value"] = p_value_male_post

print("pre:", p_value_pre)
print("post:", p_value_post)

print("female_pre:", p_value_female_pre)
print("female_post:", p_value_female_post)

print("male_pre:", p_value_male_pre)
print("male_post:", p_value_male_post)


max_value_pre = max(df_control_pre[plot_target].max(),df_disorder_pre[plot_target].max())
max_value_post = max(df_control_post[plot_target].max(),df_disorder_post[plot_target].max())

df_yaxis_max.loc["pre-episode", f"{plot_target}_max_value"] = max_value_pre
df_yaxis_max.loc["post-episode", f"{plot_target}_max_value"] = max_value_post

print("===== Done! End =====")
embed(globals(), locals())
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
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=16)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=4, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array
# print("===== Done! End =====")
# embed(globals(), locals())
###############################################################################
###############################################################################
xtick_labels = ['Pre-condition', 'Post-condition']

palette_control = sns.color_palette("Paired")
palette_disorder = sns.color_palette("PiYG")
custome_palette = [palette_control[1], palette_disorder[0]]

# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(data=df, x='episode', y=f"{plot_target}", hue='treatment', palette=custome_palette, linewidth=3)
ax.legend().set_visible(False)
ax.set_xlabel(" ", fontsize=30, fontweight="bold")
# Setting the xtick labels
ax.set_xticklabels(xtick_labels, size=25, weight='bold')
if plot_target == "hgs":
    ax.set_ylabel("Raw HGS", fontsize=30, fontweight="bold")
elif plot_target == "hgs_corrected_predicted":
    ax.set_ylabel("Adjusted HGS", fontsize=30, fontweight="bold")

xticks_positios_array = add_median_labels(ax)

for x_box_pos in np.arange(0,4,2):
    if x_box_pos == 0:
        idx = "pre-episode"
    if x_box_pos == 2:
        idx = "post-episode"
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, f"{plot_target}_max_value"]-1.6, 2, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
    ax.text((x1+x2)*.5, y+h, f"p={ttest.loc[idx, f'{plot_target}_p_value']:.3f}", ha='center', va='bottom', fontsize=18, weight='bold',  color=col)

ax.set_yticks(range(0, 141, 20))
ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')
plt.ylim(0, 140)


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig(f"matched_samples_comparison_{population}_{plot_target}.png")
plt.close()

print("===== Done! End =====")
embed(globals(), locals())