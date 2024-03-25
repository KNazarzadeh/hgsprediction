import sys
import numpy as np
import pandas as pd
import math
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
from scipy import stats


from hgsprediction.load_results.load_disorder_anova_results import load_disorder_anova_results

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
session = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
disorder_cohort = sys.argv[9]
visit_session = sys.argv[10]
n_samples = sys.argv[11]
target = sys.argv[12]
anova_target = sys.argv[13]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
    
##############################################################################
df, df_anova_result, df_post_hoc_result_without_gender, df_post_hoc_result_with_gender =  load_disorder_anova_results(
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
    anova_target,
)

df.loc[df["disorder_episode"].str.contains("pre"), "episode"] = "Pre-episode"
df.loc[df["disorder_episode"].str.contains("post"), "episode"] = "Post-episode"


df_disorder = df[df["treatment"] == f"{population}"]
df_control = df[df["treatment"] == "control"]

df_pre_disorder = df_disorder[df_disorder['disorder_episode']==f"pre-{population}"]
df_post_disorder = df_disorder[df_disorder['disorder_episode']==f"post-{population}"]

df_pre_control = df_control[df_control['disorder_episode']=="pre-control"]
df_post_control= df_control[df_control['disorder_episode']=="post-control"]

# Assuming df_post_control and df_pre_control are defined elsewhere
df_interaction_control = pd.DataFrame()
df_interaction_disorder = pd.DataFrame()

# Assuming df_post_control and df_pre_control have the same indices
df_interaction_control[f"interaction_{anova_target}"] = df_post_control[f"{anova_target}"].values - df_pre_control[f"{anova_target}"].values
df_interaction_control["treatment"] = "control"
df_interaction_control["episode"] = "Interaction"

# Assuming df_post_disorder and df_pre_disorder are defined elsewhere
df_interaction_disorder[f"interaction_{anova_target}"] = df_post_disorder[f"{anova_target}"].values - df_pre_disorder[f"{anova_target}"].values
df_interaction_disorder["treatment"] = f"{population}"
df_interaction_disorder["episode"] = "Interaction"

# Concatenating the two DataFrames
df_interaction = pd.concat([df_interaction_control, df_interaction_disorder], axis=0)

print("df_interaction_control MIN=", df_pre_control["hgs_corrected_delta"].min())
print("df_interaction_control MAX=", df_pre_control["hgs_corrected_delta"].max())
print("df_interaction_disorder MIN=", df_post_control["hgs_corrected_delta"].min())
print("df_interaction_disorder MAX=", df_post_control["hgs_corrected_delta"].max())

###############################################################################
df_ranksum = pd.DataFrame(index=["pre-episode", "post-episode", "interaction"])
df_yaxis_max = pd.DataFrame(index=["pre-episode", "post-episode", "interaction"])

stat_pre, p_value_pre = ranksums(df_pre_control[f"{anova_target}"], df_pre_disorder[f"{anova_target}"])
stat_post, p_value_post = ranksums(df_post_control[f"{anova_target}"], df_post_disorder[f"{anova_target}"])

df_ranksum.loc["pre-episode", f"{anova_target}_p_value"] = p_value_pre
df_ranksum.loc["post-episode", f"{anova_target}_p_value"] = p_value_post

t_statistic, p_value_interaction = stats.ttest_ind(df_interaction_control[f"interaction_{anova_target}"], df_interaction_disorder[f"interaction_{anova_target}"])


max_value_pre = max(df_pre_control[f"{anova_target}"].max(), df_pre_disorder[f"{anova_target}"].max())
max_value_post = max(df_post_control[f"{anova_target}"].max(), df_post_disorder[f"{anova_target}"].max())
max_value_interaction = max(df_interaction_control[f"interaction_{anova_target}"].max(), df_interaction_disorder[f"interaction_{anova_target}"].max())


df_yaxis_max.loc["pre-episode", f"{anova_target}_max_value"] = max_value_pre
df_yaxis_max.loc["post-episode", f"{anova_target}_max_value"] = max_value_post
df_yaxis_max.loc["interaction", f"{anova_target}_max_value"] = max_value_interaction


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
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=20)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=4, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array


###############################################################################

palette_control = sns.color_palette("Paired")
palette_disorder = sns.color_palette("PiYG")
custome_palette = [palette_control[1], palette_disorder[0]]

# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(data=df_interaction, x='episode', y=f"interaction_{anova_target}", hue='treatment', palette=custome_palette, linewidth=3)
ax.legend().set_visible(False)
ax.set_xlabel("Interaction", fontsize=25, fontweight="bold")

ax.set_xticklabels("")
if anova_target == "hgs":
    ax.set_ylabel("Raw HGS", fontsize=30, fontweight="bold")
elif anova_target == "hgs_corrected_delta":
    ax.set_ylabel("Adjusted HGS", fontsize=30, fontweight="bold")

xticks_positios_array = add_median_labels(ax)

x_box_pos = 0
x1 = xticks_positios_array[x_box_pos]
x2 = xticks_positios_array[x_box_pos+1]
y, h, col = df_yaxis_max.loc["interaction", f"{anova_target}_max_value"]+1, 2, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c=col)
ax.text((x1+x2)*.5, y+h, f"p={p_value_interaction:.3f}", ha='center', va='bottom', fontsize=18, weight='bold',  color=col)

ax.set_ylim(ymin=-20, ymax=40)
ax.set_yticks(range(-20, 41, 10))
ax.set_yticklabels(ax.get_yticks(), size=20, weight='bold')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig(f"CRC_{population}_{anova_target}_interaction.png")
plt.close()
print("===== Done! End =====")
embed(globals(), locals())