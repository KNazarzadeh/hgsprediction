import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_results import healthy
import statsmodels.api as sm

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]

df_combined = pd.DataFrame()
for target in ["hgs_left", "hgs_right", "hgs_L+R"]:
    df_2 = healthy.load_hgs_predicted_results(population,
        mri_status,
        model_name,
        feature_type,
        target,
        "both_gender",
        session="2",
    )

    df_3 = healthy.load_hgs_predicted_results(population,
        mri_status,
        model_name,
        feature_type,
        target,
        "both_gender",
        session="3",
    )
    print("===== Done! =====")
    embed(globals(), locals())
    df_2_intersection = df_2[df_2.index.isin(df_3.index)]
    df_3_intersection = df_3[df_3.index.isin(df_2.index)]

    target_string = " ".join([word.upper() for word in target.split("_")])
    df_2_intersection.loc[:, "target"] = f"{target_string}"
    df_3_intersection.loc[:, "target"] = f"{target_string}"
    
    df_2_intersection.loc[:, "scan_session"] = "1st scan session"
    df_3_intersection.loc[:, "scan_session"] = "2nd scan ession"
    
    selected_cols_2 = [col for col in df_2_intersection.columns if any(item in col for item in ["gender", "actual", "predicted", "(actual-predicted)", "scan_session", "target"])]
    df_selected_2 = df_2_intersection[selected_cols_2]
    
    selected_cols_3 = [col for col in df_3_intersection.columns if any(item in col for item in ["gender", "actual", "predicted", "(actual-predicted)", "scan_session", "target"])]
    df_selected_3 = df_3_intersection[selected_cols_3]
    
    prefix = f"1st_scan_{target}_"
    df_selected_2.columns = df_selected_2.columns.str.replace(prefix, '')
    prefix = f"2nd_scan_{target}_"
    df_selected_3.columns = df_selected_3.columns.str.replace(prefix, '')
    
    df_merged = pd.concat([df_selected_2, df_selected_3]) 
    df_combined = pd.concat([df_combined, df_merged])

###############################################################################
def add_median_labels(ax, fmt='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=10)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

###############################################################################
feature_string = " + ".join([word.capitalize() for word in feature_type.split("_")])
###############################################################################
df_combined['gender'] = df_combined["gender"].map({0: 'female', 1: 'male'})
df_combined.loc[:, "scan_session_gender"] = df_combined["scan_session"] + " " + df_combined["gender"]
df_combined.loc[:, "target_gender"] = df_combined["target"] + " " + df_combined["gender"]

# melted_data = pd.melt(df_combined, id_vars=['gender', 'scan_session', 'target'], var_name='hgs_type', value_name='hgs_value')
# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#a851ab', '#005c95'])  # You can use any hex color codes you prefer
# Create the boxplot for 'hgs_category' and 'gender'
# Create the boxplot with the custom palette
for index, yaxis_target in enumerate(["actual", "predicted", "(actual-predicted)"]):
    if yaxis_target == "(actual-predicted)":
        yaxis_target_string = "Prediction error"
    else:
        yaxis_target_string = yaxis_target.capitalize()    
    fig = plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="target_gender", y=yaxis_target, data=df_combined, hue="scan_session", palette=custom_palette)
    
    # Select which box you want to change    
    ax.patches[1].set_facecolor('#a851ab')
    ax.patches[3].set_facecolor('#a851ab')
    ax.patches[4].set_facecolor('#005c95')
    ax.patches[5].set_facecolor('#005c95')
    ax.patches[6].set_facecolor('#a851ab')
    ax.patches[7].set_facecolor('#a851ab')
    ax.patches[8].set_facecolor('#005c95')
    ax.patches[9].set_facecolor('#005c95')
    ax.patches[10].set_facecolor('#a851ab')
    ax.patches[11].set_facecolor('#a851ab')
    ax.patches[12].set_facecolor('#005c95')
    ax.patches[13].set_facecolor('#005c95')
    
    
    ymin , ymax = ax.get_ylim()
    # changing the fontsize of ticks
    plt.yticks(np.arange(min(ax.get_yticks()), max(ax.get_yticks()), 10))

    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    if yaxis_target_string == "Prediction error":
        plt.ylabel(f"{yaxis_target_string}", fontsize=20, fontweight="bold")
    else:
        plt.ylabel(f"{yaxis_target_string} HGS", fontsize=20, fontweight="bold")
    plt.title(f"HGS for 1st and 2nd MRI scans - Healthy controls\n{feature_string}", fontsize=20, fontweight="bold")
    
    # Get the handles and labels from the legend
    handles, labels = ax.get_legend_handles_labels()

    # Remove the last two items (handles and labels)
    # handles = handles[:-2]
    # labels = labels[:-2]

    # Set the modified handles and labels back to the legend
    legend = ax.legend(handles, labels)
    
    # # Modify individual legend labels
    # legend = plt.legend(title="Gender", loc="upper left")  # Add legend
    legend.get_texts()[0].set_text(f"Female: N={len(df_2_intersection[df_2_intersection['gender']==0])}")
    legend.get_texts()[1].set_text(f"Male: N={len(df_2_intersection[df_2_intersection['gender']==1])}")
    
    # Show the plot
    plt.tight_layout()

    add_median_labels(ax)
    # medians = melted_df.groupby(['combine_hgs_stroke_cohort_category', 'gender'])['value'].median()

    plt.show()
    plt.savefig(f"femalemale_nexttoeachother_{population}_{yaxis_target}_hgs_female_male_separated.png")
    plt.close()
print("===== Done! =====")
embed(globals(), locals())
###############################################################################

print("===== Done! =====")
embed(globals(), locals())


