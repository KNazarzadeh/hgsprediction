import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import stroke

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_longitudinal = stroke.load_hgs_predicted_results(population, mri_status, session_column, model_name, feature_type, target, "both_gender")
    
selected_cols = [col for col in df_longitudinal.columns if any(item in col for item in ["actual", "predicted"])]
df_selected = df_longitudinal[selected_cols]

df_selected.insert(0, "gender", df_longitudinal["gender"])

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
 # Create the boxplot with the custom palette
fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # Adjust the figure size if needed
sns.set(style="whitegrid")
# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer
# Create a list of column groups
for index, yaxis_target in enumerate(["actual", "predicted"]):
    column_groups = [[f"1st_pre-stroke_{target}_{yaxis_target}", f"1st_post-stroke_{target}_{yaxis_target}"]]
    # df = df_selected[column_groups]
    # df.columns = ["Pre-stroke", "post_stroke"]
    # Initialize an empty list to store the melted DataFrames
    melted_dfs = []
    # Iterate through column groups and create melted DataFrames
    for group_columns in column_groups:
        # Melt the DataFrame for the current group
        melted_group = pd.melt(df_selected, id_vars=["gender"], value_vars=group_columns, var_name="variable", ignore_index=False)        
        # Create 'stroke_cohort' based on 'variable'        
        melted_group['stroke_cohort'] = melted_group['variable'].apply(lambda x: 'Pre-stroke' if 'pre-stroke' in x else ('Post-stroke' if 'post-stroke' in x else None))
        
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
    sns.boxplot(x="stroke_cohort", y="value", data=melted_df, palette=custom_palette, ax=ax[index])
    stripplot = sns.stripplot(x="stroke_cohort", y="value", data=melted_df, color='black', jitter=False, linewidth=1, ax=ax[index])
    lineplot = sns.lineplot(data=melted_df, x="stroke_cohort", y="value",  estimator=None, units="SubjectID", markers=True, color="grey", linewidth=1, ax=ax[index])
    print(index, yaxis_target)
    print(column_groups)
    
    if yaxis_target == "actual":
        ax[index].set_title(f"Actual HGS", fontsize=20, fontweight="bold")
    elif yaxis_target == "predicted":
        ax[1].set_title("Predicted HGS", fontsize=20, fontweight="bold")

fig.suptitle(f"Title -  (N={len(df_longitudinal)})\nTarget={target}, Features=Anthropometrics and Age", fontsize="14", fontweight="bold")

fig.text(0.04, 0.5, 'HGS Values', va='center', rotation='vertical', fontsize=20, fontweight="bold")
ax[0].set_xlabel("")
ax[0].set_ylabel("")
ax[1].set_xlabel("")
ax[1].set_ylabel("")

xmin = min(ax[0].get_xlim()[0], ax[1].get_xlim()[0])
xmax = max(ax[0].get_xlim()[1], ax[1].get_xlim()[1])
ymin = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
ymax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])

ax[0].set_xlim(xmin, xmax)
ax[1].set_xlim(xmin, xmax)
ax[0].set_ylim(ymin, ymax)
ax[1].set_ylim(ymin, ymax)

# legend0 = ax[0].legend(title="Gender", loc="upper right")  # Add legend
# legend1= ax[1].legend(title="Gender", loc="upper right")  # Add legend

# Modify individual legend labels
female_n = len(df_longitudinal[df_longitudinal["gender"]==0])
male_n = len(df_longitudinal[df_longitudinal["gender"]==1])

# legend0.get_texts()[0].set_text(f"Female: N={female_n}")
# legend0.get_texts()[1].set_text(f"Male: N={male_n}")
# legend1.get_texts()[0].set_text(f"Female: N={female_n}")
# legend1.get_texts()[1].set_text(f"Male: N={male_n}")

plt.show()
plt.savefig("hh_right.png")
print("===== Done! =====")
embed(globals(), locals())
    # # Add labels and title
    # plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    # plt.ylabel(f"{yaxis_target.capitalize()} HGS values", fontsize=20, fontweight="bold")
    # plt.title(f"pre-stroke and post-stroke HGS values - {feature_type}", fontsize=20)

    # ymin , ymax = ax.get_ylim()
    # if yaxis_target == "(actual-predicted)":
    #     plt.yticks(np.arange(-40, 50, 10))
    # else:
    #     plt.yticks(np.arange(0, 120, 10))
    # # Show the plot
    # plt.legend(title="stroke cohort", loc="upper left")  # Add legend
    # legend = plt.legend(title="Gender", loc="upper left")  # Add legend
    # # Modify individual legend labels
    # legend.get_texts()[0].set_text(f"Pre-stroke: N={len(df_longitudinal)}")
    # legend.get_texts()[1].set_text(f"Post-stroke: N={len(df_longitudinal)}")

    # plt.tight_layout()

    # add_median_labels(ax)
    # # medians = melted_df.groupby(['hgs_category', 'stroke_cohort'])['value'].median()
    # plt.show()
    # plt.savefig(f"pair_plot.png")
    # plt.close()
    ###############################################################################
    # # Define a custom palette with two blue colors
    # custom_palette = sns.color_palette(['#800080', '#000080'])  # You can use any hex color codes you prefer
    # # Create the boxplot for 'hgs_category' and 'gender'
    # plt.figure(figsize=(12, 6))
    # sns.set(style="whitegrid")
    # ax = sns.boxplot(x="combine_hgs_stroke_cohort_category", y="value", data=melted_df, hue="gender", palette=custom_palette)
    # if yaxis_target == "(actual-predicted)":
    #     plt.yticks(np.arange(-40, 50, 10))
    # else:
    #     plt.yticks(np.arange(0, 120, 10))
    # # # Extract unique values of 'hgs_category' and 'stroke_cohort'
    # unique_hgs_categories = melted_df["hgs_category"].unique()
    # unique_stroke_cohorts = melted_df["stroke_cohort"].unique()

    # # Create xtick labels by repeating the 'stroke_cohort' array for each 'hgs_category'
    # xticks_labels = [f"{cohort}" for hgs_category in unique_hgs_categories for cohort in unique_stroke_cohorts]
    # # Set xtick positions and labels
    # plt.xticks(range(len(xticks_labels)), xticks_labels)

    # # Add labels and title
    # plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    # plt.ylabel(f"{yaxis_target.capitalize()} HGS values", fontsize=20, fontweight="bold")
    # plt.title(f"HGS values for genders - {feature_type}", fontsize=20)
    # legend = plt.legend(title="Gender", loc="upper left")  # Add legend
    # # Modify individual legend labels
    # legend.get_texts()[0].set_text(f"Female: N={len(df_longitudinal[df_longitudinal['gender']==0])}")
    # legend.get_texts()[1].set_text(f"Male: N={len(df_longitudinal[df_longitudinal['gender']==1])}")

    # # Show the plot
    # plt.tight_layout()

    # add_median_labels(ax)
    # # medians = melted_df.groupby(['combine_hgs_stroke_cohort_category', 'gender'])['value'].median()

    # plt.show()
    # plt.savefig(f"{yaxis_target}_hgs_female_male_separated.png")
    # plt.close()



print("===== Done! =====")
embed(globals(), locals())


