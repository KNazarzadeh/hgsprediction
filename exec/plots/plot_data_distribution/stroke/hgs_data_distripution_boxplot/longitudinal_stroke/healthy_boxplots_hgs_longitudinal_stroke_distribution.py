import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import stroke_load_data
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
    
    df_2_intersection = df_2[df_2.index.isin(df_3.index)]
    df_3_intersection = df_3[df_3.index.isin(df_2.index)]
    
    df_2_intersection.loc[:, "target"] = f"{target}"
    df_3_intersection.loc[:, "target"] = f"{target}"
    
    df_2_intersection.loc[:, "scan_session"] = "1st scan session"
    df_3_intersection.loc[:, "scan_session"] = "2nd scan ession"
    
    selected_cols_2 = [col for col in df_2_intersection.columns if any(item in col for item in ["gender", "actual", "predicted", "(actual-predicted)", "scan_session", "target"])]
    df_selected_2 = df_2_intersection[selected_cols_2]
    
    selected_cols_3 = [col for col in df_3_intersection.columns if any(item in col for item in ["gender", "actual", "predicted", "(actual-predicted)", "scan_session", "target"])]
    df_selected_3 = df_3_intersection[selected_cols_3]
    
    prefix = f"{target}_"
    df_selected_2.columns = df_selected_2.columns.str.replace(prefix, '')
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
# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer

# Create the boxplot with the custom palette
for index, yaxis_target in enumerate(["actual", "predicted", "(actual-predicted)"]):
    fig = plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    # sns.set_context("poster", font_scale=1.25)
    ax = sns.boxplot(x="target", y=yaxis_target, hue="scan_session", data=df_combined, palette=custom_palette)    
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"{yaxis_target.capitalize()} HGS values", fontsize=20, fontweight="bold")
    plt.title(f"1st vs 2nd MRI scan HGS values - {feature_type}", fontsize=20)

    ymin , ymax = ax.get_ylim()
    # changing the fontsize of ticks
    plt.yticks(np.arange(min(ax.get_yticks()), max(ax.get_yticks()), 10))
    # Show the plot
    plt.legend(title="Scan sessions", loc="upper left")  # Add legend
    legend = plt.legend(title="Scan sessions", loc="upper left")  # Add legend
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"1st scan session: N={len(df_2_intersection)}")
    legend.get_texts()[1].set_text(f"2nd scan session: N={len(df_3_intersection)}")

    plt.tight_layout()

    add_median_labels(ax)
    # medians = melted_df.groupby(['hgs_category', 'stroke_cohort'])['value'].median()
    plt.show()
    plt.savefig(f"{population}_{yaxis_target}_hgs_both_gender.png")
    plt.close()

###############################################################################
df_combined['gender'] = df_combined["gender"].map({0: 'female', 1: 'male'})
df_combined.loc[:, "scan_session_gender"] = df_combined["scan_session"] + "_" + df_combined["gender"]
# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#800080', '#000080'])  # You can use any hex color codes you prefer
# Create the boxplot for 'hgs_category' and 'gender'
# Create the boxplot with the custom palette
for index, yaxis_target in enumerate(["actual", "predicted", "(actual-predicted)"]):
    fig = plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="target", y=yaxis_target, data=df_combined, hue="scan_session_gender", palette=custom_palette)
    ymin , ymax = ax.get_ylim()
    # changing the fontsize of ticks
    plt.yticks(np.arange(min(ax.get_yticks()), max(ax.get_yticks()), 10))
    # # Extract unique values of 'hgs_category' and 'stroke_cohort'
    # unique_hgs_categories = df_combined["target"].unique()
    # unique_scan_session = df_combined["scan_session"].unique()

    # # Create xtick labels by repeating the 'stroke_cohort' array for each 'hgs_category'
    # xticks_labels = [f"{cohort}" for hgs_category in unique_hgs_categories for cohort in unique_scan_session]
    # # Set xtick positions and labels
    # plt.xticks(range(len(xticks_labels)), xticks_labels)

    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"{yaxis_target.capitalize()} HGS values", fontsize=20, fontweight="bold")
    plt.title(f"HGS values for genders - {feature_type}", fontsize=20)
    legend = plt.legend(title="Gender", loc="upper left")  # Add legend
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"Female: N={len(df_2_intersection[df_2_intersection['gender']==0])}")
    legend.get_texts()[1].set_text(f"Male: N={len(df_2_intersection[df_2_intersection['gender']==1])}")

    # Show the plot
    plt.tight_layout()

    add_median_labels(ax)
    # medians = melted_df.groupby(['combine_hgs_stroke_cohort_category', 'gender'])['value'].median()

    plt.show()
    plt.savefig(f"{population}_{yaxis_target}_hgs_female_male_separated.png")
    plt.close()
print("===== Done! =====")
embed(globals(), locals())
###############################################################################

print("===== Done! =====")
embed(globals(), locals())


