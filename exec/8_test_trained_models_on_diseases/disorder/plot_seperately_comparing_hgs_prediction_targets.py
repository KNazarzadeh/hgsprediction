import sys
import numpy as np
import pandas as pd
import math
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects


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
anova_target = sys.argv[12]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

y_hgs = anova_target
    
##############################################################################

df = pd.DataFrame()
for target in ["hgs_left", "hgs_right", "hgs_L+R"]:
    df_data, df_anova_result, df_post_hoc_result_without_gender, df_post_hoc_result_with_gender =  load_disorder_anova_results(
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
    
    df = pd.concat([df, df_data], axis=0)
    
df_pre = df[df["disorder_episode"].str.startswith("pre")]
df_post = df[df["disorder_episode"].str.startswith("post")]

df_pre_disorder = df_pre[df_pre['treatment']==f"{population}"]
df_post_disorder = df_post[df_post['treatment']==f"{population}"]

df_control_pre = df_pre[df_pre['treatment']=="control"]
df_control_post= df_post[df_post['treatment']=="control"]

###############################################################################
df_ranksum = pd.DataFrame(index=["hgs_left", "hgs_right", "hgs_L+R"])
df_yaxis_max = pd.DataFrame(index=["hgs_left", "hgs_right", "hgs_L+R"])

for target in ["hgs_left", "hgs_right", "hgs_L+R"]:  
    stat_pre, p_value_pre = ranksums(df_control_pre[df_control_pre["hgs_target"]==target][y_hgs], df_pre_disorder[df_pre_disorder["hgs_target"]==target][y_hgs])
    stat_post, p_value_post = ranksums(df_control_post[df_control_post["hgs_target"]==target][y_hgs], df_post_disorder[df_post_disorder["hgs_target"]==target][y_hgs])
    
    df_ranksum.loc[target, f"pre_{y_hgs}_p_value"] = p_value_pre
    df_ranksum.loc[target, f"pre_{y_hgs}_stat_value"] = stat_pre
    df_ranksum.loc[target, f"post_{y_hgs}_p_value"] = p_value_post
    df_ranksum.loc[target, f"post_{y_hgs}_stat_value"] = stat_post

    max_value_pre = max(df_control_pre[df_control_pre["hgs_target"] == target][y_hgs].max(),df_pre_disorder[df_pre_disorder["hgs_target"] == target][y_hgs].max())
    max_value_post = max(df_control_post[df_control_post["hgs_target"] == target][y_hgs].max(),df_post_disorder[df_post_disorder["hgs_target"] == target][y_hgs].max())

    df_yaxis_max.loc[target, f"pre_{y_hgs}_max_value"] = max_value_pre
    df_yaxis_max.loc[target, f"post_{y_hgs}_max_value"] = max_value_post
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
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=16)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=4, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array

###############################################################################
xtick_labels = ['Left HGS', 'Right HGS', 'Combined HGS']

palette_control = sns.color_palette("Paired")
palette_disorder = sns.color_palette("PiYG")
custome_palette = [palette_control[1], palette_disorder[0]]

ymin = 0
ymax =0


# Set the style of seaborn
sns.set_style("whitegrid")
# Create the boxplot
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0]
sns.boxplot(data=df_pre, x='hgs_target', y=f'{y_hgs}', hue='treatment', palette=custome_palette, ax=ax[0])
ax[0].legend().set_visible(False)
ax[0].set_xlabel("HGS target", fontsize=30, fontweight="bold")
# Setting the xtick labels
ax[0].set_xticklabels(xtick_labels, size=20, weight='bold')
ax[0].set_title(f"Pre-episode - {population}(N={int(len(df_pre[df_pre['treatment']==population])/3)}) - matched controls(N={int(len(df_pre[df_pre['treatment']=='control'])/3)})", fontsize=12, fontweight="bold")            
if y_hgs == "hgs_delta":
    ax[0].set_ylabel("Delta HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs_corrected_delta":
    ax[0].set_ylabel("Corrected delta HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs_predicted":
    ax[0].set_ylabel("Predicted HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs_corrected_predicted":
    ax[0].set_ylabel("Corrected predicted HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs":
    ax[0].set_ylabel("True HGS", fontsize=30, fontweight="bold")
    
xticks_positios_array = add_median_labels(ax[0])
for x_box_pos in np.arange(0,6,2):
    if x_box_pos == 0:
        idx = "hgs_left"
    if x_box_pos == 2:
        idx = "hgs_right"
    if x_box_pos == 4:
        idx = "hgs_L+R"
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, f"pre_{y_hgs}_max_value"] + 1, 2, 'k'
    ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax[0].text((x1+x2)*.5, y+h, f"p={df_ranksum.loc[idx, f'pre_{y_hgs}_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold',  color=col)
ymin_tmp, ymax_tmp = ax[0].get_ylim()

# if ymin_tmp < ymin:
#     ymin = ymin_tmp
# if ymax_tmp > ymax:
#     ymax = ymax_tmp

sns.boxplot(data=df_post, x='hgs_target', y=f'{y_hgs}', hue='treatment', palette=custome_palette, ax=ax[1])
ax[1].legend().set_visible(False)
ax[1].set_xlabel("HGS target", fontsize=30, fontweight="bold")

# Setting the xtick labels
ax[1].set_xticklabels(xtick_labels, size=20, weight='bold')       
ax[1].set_title(f"Post-episode - {population}(N={int(len(df_pre[df_pre['treatment']==population])/3)})- matched controls(N={int(len(df_pre[df_pre['treatment']=='control'])/3)})", fontsize=12, fontweight="bold")                                
if y_hgs == "hgs_delta":
    ax[1].set_ylabel("Delta HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs_corrected_delta":
    ax[1].set_ylabel("Corrected delta HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs_predicted":
    ax[1].set_ylabel("Predicted HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs_corrected_predicted":
    ax[1].set_ylabel("Corrected predicted HGS", fontsize=30, fontweight="bold")
elif y_hgs == "hgs":
    ax[1].set_ylabel("True HGS", fontsize=30, fontweight="bold")

xticks_positios_array = add_median_labels(ax[1])

for x_box_pos in np.arange(0,6,2):
    if x_box_pos == 0:
        idx = "hgs_left"
    if x_box_pos == 2:
        idx = "hgs_right"
    if x_box_pos == 4:
        idx = "hgs_L+R"
    x1 = xticks_positios_array[x_box_pos]
    x2 = xticks_positios_array[x_box_pos+1]
    y, h, col = df_yaxis_max.loc[idx, f"post_{y_hgs}_max_value"]+ 2, 2, 'k'
    ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax[1].text((x1+x2)*.5, y+h, f"p={df_ranksum.loc[idx, f'post_{y_hgs}_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold',  color=col)
    
# if ymin_tmp < ymin:
#     ymin = ymin_tmp
# if ymax_tmp > ymax:
#     ymax = ymax_tmp

# if population == "depression":
#     ylim_range = range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 20) 
# if population == "parkinson":
#     ylim_range = range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+20, 20) 
# if population == "stroke":
#     ylim_range = range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+40, 20) 

# ax[1].set_ylim(min(ylim_range), max(ylim_range))
# ax[1].set_yticks(range(min(ylim_range), max(ylim_range)+1, 20))
# ax[1].set_yticklabels(ax[1].get_yticks(), size=25, weight='bold')
# ax[0].set_ylim(min(ylim_range), max(ylim_range))
# ax[0].set_yticks(range(min(ylim_range), max(ylim_range)+1, 20))
# ax[0].set_yticklabels(ax[0].get_yticks(), size=25, weight='bold')


ax[1].set_ylim(0, 151)
ax[1].set_yticks(range(0, 150+10, 20))
ax[1].set_yticklabels(ax[1].get_yticks(), size=20, weight='bold')
ax[0].set_ylim(0, 150)
ax[0].set_yticks(range(0, 150+10, 20))
ax[0].set_yticklabels(ax[0].get_yticks(), size=20, weight='bold')


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig(f"CRC_Beheshti_{population}_{y_hgs}_corrected_predictions_matched_controls.png")
plt.close()

print("===== Done! End =====")
embed(globals(), locals())