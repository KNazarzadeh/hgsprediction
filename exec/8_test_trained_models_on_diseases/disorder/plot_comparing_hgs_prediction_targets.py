import sys
import numpy as np
import pandas as pd
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
if anova_target == "delta":
    plot_target = ["hgs_delta", "hgs_corrected_delta"]

elif anova_target == "predicted":
    plot_target = ["hgs_predicted", "hgs_corrected_predicted"]

##############################################################################
for i, anova_target in enumerate(plot_target):
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
        print("===== Done! =====")
        embed(globals(), locals())
    df["anova_target"] = anova_target
    
    print("===== Done! =====")
    embed(globals(), locals())

    df_pre_episode = df[df["disorder_episode"].str.startswith("pre")]
    df_post_episode = df[df["disorder_episode"].str.startswith("post")]

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
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=12)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array
###############################################################################
# Create the boxplot
fig, ax = plt.figure(figsize=(12, 8))

# Set the style of seaborn
sns.set(style="whitegrid")

sns.boxplot(data=df_pre_episode, x='hgs_target', y='hgs_delta', hue='treatment', palette='Set3')

# Add title
plt.suptitle('Boxplots of hgs by Disorder Episode, hgs_target, Treatment, and Gender')

xticks_positios_array = add_median_labels(ax)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig("x.png")
plt.close()




stat, p_value = ranksums(tmp_samples["value"], tmp_depression["value"])


