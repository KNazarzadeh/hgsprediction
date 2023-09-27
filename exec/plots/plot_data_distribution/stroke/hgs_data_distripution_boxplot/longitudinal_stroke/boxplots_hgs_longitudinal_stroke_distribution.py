import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import stroke
import statsmodels.api as sm

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]

stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_combined = pd.DataFrame()
for target in ["hgs_L+R", "hgs_left", "hgs_right"]:
    df_longitudinal = stroke.load_hgs_predicted_results(population, mri_status, session_column, model_name, feature_type, target, "both_gender")
    
    selected_cols = [col for col in df_longitudinal.columns if any(item in col for item in ["actual", "predicted", "(actual-predicted)"])]
    df_selected = df_longitudinal[selected_cols]
    
    df_combined = pd.concat([df_combined, df_selected], axis=1)    

df_combined.insert(0, "gender", df_longitudinal["gender"])

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
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Create a list of column groups
for yaxis_target in ["actual", "predicted", "(actual-predicted)"]:
    column_groups = [
    ("HGS Left", [f"1st_pre-stroke_hgs_left_{yaxis_target}", f"1st_post-stroke_hgs_left_{yaxis_target}"]),
    ("HGS Right", [f"1st_pre-stroke_hgs_right_{yaxis_target}", f"1st_post-stroke_hgs_right_{yaxis_target}"]),
    ("HGS L+R", [f"1st_pre-stroke_hgs_L+R_{yaxis_target}", f"1st_post-stroke_hgs_L+R_{yaxis_target}"])
    ]

    # Initialize an empty list to store the melted DataFrames
    melted_dfs = []
    # Iterate through column groups and create melted DataFrames
    for group_name, group_columns in column_groups:
        # Melt the DataFrame for the current group
        melted_group = pd.melt(df_combined, id_vars=["gender"], value_vars=group_columns, var_name="variable", ignore_index=False)
        
        # Create 'hgs_category' based on 'group_name'
        melted_group['hgs_category'] = group_name
        
        # Create 'stroke_cohort' based on 'variable'        
        melted_group['stroke_cohort'] = melted_group['variable'].apply(lambda x: 'Pre-stroke' if 'pre-stroke' in x else ('Post-stroke' if 'post-stroke' in x else None))             
        # Add a new column based on the combination of 'hgs_category' and 'stroke_cohort'
        melted_group['combine_hgs_stroke_cohort_category'] = group_name + '-' + melted_group['stroke_cohort']
        
        # Create 'gender' based on 'gender' column
        melted_group['gender'] = melted_group["gender"].map({0: 'female', 1: 'male'})
        
        # Drop the original 'variable' column
        # melted_group.drop(columns=["gender"], inplace=True)
        
        # Append the melted DataFrame to the list
        melted_dfs.append(melted_group)

    # Concatenate the melted DataFrames into one without ignoring the original indexes
    melted_df = pd.concat(melted_dfs, ignore_index=False)

    print(melted_df)
    ###############################################################################
    # Define a custom palette with two blue colors
    custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer

    # Create the boxplot with the custom palette
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    # sns.set_context("poster", font_scale=1.25)
    ax = sns.boxplot(x="hgs_category", y="value", hue="stroke_cohort", data=melted_df, palette=custom_palette)    
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"{yaxis_target.capitalize()} HGS values", fontsize=20, fontweight="bold")
    plt.title(f"pre-stroke and post-stroke HGS values - {feature_type}", fontsize=20)

    ymin , ymax = ax.get_ylim()
    if yaxis_target == "(actual-predicted)":
        plt.yticks(np.arange(-40, 50, 10))
    else:
        plt.yticks(np.arange(0, 120, 10))
    # Show the plot
    plt.legend(title="stroke cohort", loc="upper left")  # Add legend
    legend = plt.legend(title="Stroke cohort", loc="upper left")  # Add legend
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"Pre-stroke: N={len(df_longitudinal)}")
    legend.get_texts()[1].set_text(f"Post-stroke: N={len(df_longitudinal)}")

    plt.tight_layout()

    add_median_labels(ax)
    # medians = melted_df.groupby(['hgs_category', 'stroke_cohort'])['value'].median()
    plt.show()
    plt.savefig(f"{population}_{feature_type}_{yaxis_target}_hgs_both_gender.png")
    plt.close()
###############################################################################
###############################################################################
# Create a list of column groups
for yaxis_target in ["actual", "predicted", "(actual-predicted)"]:
    column_groups = [
    ("HGS Left", [f"1st_pre-stroke_hgs_left_{yaxis_target}", f"1st_post-stroke_hgs_left_{yaxis_target}"]),
    ("HGS Right", [f"1st_pre-stroke_hgs_right_{yaxis_target}", f"1st_post-stroke_hgs_right_{yaxis_target}"]),
    ("HGS L+R", [f"1st_pre-stroke_hgs_L+R_{yaxis_target}", f"1st_post-stroke_hgs_L+R_{yaxis_target}"])
    ]

    # Initialize an empty list to store the melted DataFrames
    melted_dfs = []
    # Iterate through column groups and create melted DataFrames
    for group_name, group_columns in column_groups:
        # Melt the DataFrame for the current group
        melted_group = pd.melt(df_combined, id_vars=["gender"], value_vars=group_columns, var_name="variable", ignore_index=False)
        
        # Create 'hgs_category' based on 'group_name'
        melted_group['hgs_category'] = group_name
        
        # Create 'stroke_cohort' based on 'variable'        
        melted_group['stroke_cohort'] = melted_group['variable'].apply(lambda x: 'Pre-stroke' if 'pre-stroke' in x else ('Post-stroke' if 'post-stroke' in x else None))             
        # Add a new column based on the combination of 'hgs_category' and 'stroke_cohort'
        melted_group['combine_hgs_stroke_cohort_category'] = group_name + '-' + melted_group['stroke_cohort']

        # Create 'gender' based on 'gender' column
        melted_group['gender'] = melted_group["gender"].map({0: 'female', 1: 'male'})
        melted_group['combine_hgs_gender'] = group_name + '-' + melted_group['gender']
        # Drop the original 'variable' column
        # melted_group.drop(columns=["gender"], inplace=True)
        
        # Append the melted DataFrame to the list
        melted_dfs.append(melted_group)

    # Concatenate the melted DataFrames into one without ignoring the original indexes
    melted_df = pd.concat(melted_dfs, ignore_index=False)

    print(melted_df)
    ###############################################################################
    # Define a custom palette with two blue colors
    # Create a custom palette dictionary based on the values of the x-axis variable
    # custom_palette = sns.color_palette(['#800080', '#000080'])  # You can use any hex color codes you prefer
    custom_palette = {
    ('HGS Left-female', 'Pre-stroke'): '#800080',
    ('HGS Left-female', 'Post-stroke'): '#800080',
    ('HGS Right-female', 'Pre-stroke'): '#800080',
    ('HGS Right-female', 'Post-stroke'): '#800080',        
    ('HGS L+R-female', 'Pre-stroke'): '#800080',
    ('HGS L+R-female', 'Post-stroke'): '#800080',        
    ('HGS Left-male', 'Pre-stroke'): '#000080',
    ('HGS Left-male', 'Post-stroke'): '#000080',
    ('HGS Right-male', 'Pre-stroke'): '#000080',
    ('HGS Right-male', 'Post-stroke'): '#000080',
    ('HGS L+R-male', 'Pre-stroke'): '#000080',
    ('HGS L+R-male', 'Post-stroke'): '#000080', 
    }

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    # ax = sns.boxplot(x="combine_hgs_stroke_cohort_category", y="value", data=melted_df, hue="gender", palette=custom_palette)
    ax = sns.boxplot(x="combine_hgs_gender", y="value", data=melted_df, hue="stroke_cohort", palette=custom_palette, hue_order=melted_df["stroke_cohort"].unique())  # Ensure correct hue order)

    if yaxis_target == "(actual-predicted)":
        plt.yticks(np.arange(-40, 50, 10))
    else:
        plt.yticks(np.arange(0, 120, 10))
    # # Extract unique values of 'hgs_category' and 'stroke_cohort'
    unique_hgs_categories = melted_df["hgs_category"].unique()
    unique_stroke_cohorts = melted_df["stroke_cohort"].unique()

    # Create xtick labels by repeating the 'stroke_cohort' array for each 'hgs_category'
    xticks_labels = [f"{cohort}" for hgs_category in unique_hgs_categories for cohort in unique_stroke_cohorts]
    # Set xtick positions and labels
    plt.xticks(range(len(xticks_labels)), xticks_labels)

    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"{yaxis_target.capitalize()} HGS values", fontsize=20, fontweight="bold")
    plt.title(f"HGS values for genders - {feature_type}", fontsize=20)
    legend = plt.legend(title="Gender", loc="upper left")  # Add legend
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"Female: N={len(df_longitudinal[df_longitudinal['gender']==0])}")
    legend.get_texts()[1].set_text(f"Male: N={len(df_longitudinal[df_longitudinal['gender']==1])}")

    # Show the plot
    plt.tight_layout()

    add_median_labels(ax)
    # medians = melted_df.groupby(['combine_hgs_stroke_cohort_category', 'gender'])['value'].median()

    plt.show()
    plt.savefig("aaa.png")
print("===== Done! =====")
embed(globals(), locals())
    # plt.savefig(f"{population}_{feature_type}_{yaxis_target}_hgs_female_male_separated.png")
    # plt.close()
print("===== Done! =====")
embed(globals(), locals())
###############################################################################



