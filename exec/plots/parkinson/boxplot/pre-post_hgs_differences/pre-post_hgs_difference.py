import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import parkinson_load_data
from hgsprediction.load_results import parkinson

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]

parkinson_cohort = "longitudinal-parkinson"
session_column = f"1st_{parkinson_cohort}_session"
df_all = pd.DataFrame()
for target in ["hgs_left", "hgs_right", "hgs_L+R"]:
    df_longitudinal = parkinson.load_hgs_predicted_results(population, mri_status, session_column, model_name, feature_type, target, "both_gender")

    selected_cols = [col for col in df_longitudinal.columns if any(item in col for item in ["actual", "predicted"])]
    df_selected = df_longitudinal[selected_cols]

    df_selected.insert(0, "gender", df_longitudinal["gender"].map({0: 'female', 1: 'male'}))

    df_selected["pre-post_hgs_difference"] = df_selected[f"1st_pre-parkinson_{target}_actual"] - df_selected[f"1st_post-parkinson_{target}_actual"]
    converted_text = target.title()
    converted_text = converted_text.replace('Hgs', 'HGS')
    df_selected["hgs_target_cohort"] = converted_text

    df_all = pd.concat([df_all, df_selected[["gender", "pre-post_hgs_difference", "hgs_target_cohort"]]])

###############################################################################
def add_median_labels(ax, fmt='.0f'):
    xticks_positios_array = []
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=14)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array
###############################################################################
custom_palette = sns.color_palette(['#800080', '#000080'])  # You can use any hex color codes you prefer
# Create the boxplot with the custom palette
sns.set(style="whitegrid")
fig = plt.subplots(figsize=(15, 10))  # Adjust the figure size if needed
sns.set(style="whitegrid")
ax = sns.boxplot(x="hgs_target_cohort", y="pre-post_hgs_difference", data=df_all, hue ="gender", palette=custom_palette)

# Add labels and title
plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
plt.ylabel(f"Pre-Post HGS Values", fontsize=20, fontweight="bold")
plt.title(f"Pre- and Post- HGS diferences-{population.capitalize()}", fontsize=15, fontweight="bold")
plt.yticks(fontsize=18, weight='bold')
plt.xticks(fontsize=18, weight='bold')
legend = plt.legend(loc="upper left", prop={'size': 16, 'weight': 'bold'})
legend.set_title("Gender", {'size': 16, 'weight': 'bold'})
# Get the original legend labels
legend_labels = [text.get_text() for text in legend.get_texts()]
# Modify individual legend labels
legend.get_texts()[0].set_text(f"{legend_labels[0]}(N={len(df_longitudinal[df_longitudinal['gender']==0])})")
legend.get_texts()[1].set_text(f"{legend_labels[1]}(N={len(df_longitudinal[df_longitudinal['gender']==1])})") 
plt.tight_layout()

add_median_labels(ax)

plt.show()
plt.savefig(f"boxplot_pre-post_hgs_difference_{population}.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
