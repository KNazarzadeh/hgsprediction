import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import healthy

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

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

df_2_intersection.loc[:, "scan_session"] = "1st scan session"
df_3_intersection.loc[:, "scan_session"] = "2nd scans ession"

###############################################################################
 # Create the boxplot with the custom palette
fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # Adjust the figure size if needed
sns.set(style="whitegrid")
# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer
hue_pallete = sns.color_palette(['#DA70D6', 'blue'])  
# Create a list of column groups
for index, yaxis_target in enumerate(["actual", "predicted"]):
    df_merged = pd.concat([df_2_intersection[["gender", "scan_session", f"{target}_{yaxis_target}"]], df_3_intersection[["gender", "scan_session", f"{target}_{yaxis_target}"]]])

    ###############################################################################
    sns.boxplot(x="scan_session", y=f"{target}_{yaxis_target}", data=df_merged, palette=custom_palette, ax=ax[index])
    sns.stripplot(x="scan_session", y=f"{target}_{yaxis_target}", data=df_merged, hue="gender", palette=hue_pallete, jitter=False, linewidth=1, ax=ax[index])
    sns.lineplot(data=df_merged, x="scan_session", y=f"{target}_{yaxis_target}",  hue="gender", estimator=None, units="SubjectID", markers=True, color="grey", palette=hue_pallete, linewidth=1, legend=False, ax=ax[index])

    if yaxis_target == "actual":
        ax[index].set_title(f"Actual HGS", fontsize=20, fontweight="bold")
    elif yaxis_target == "predicted":
        ax[1].set_title("Predicted HGS", fontsize=20, fontweight="bold")

fig.suptitle(f"Actual vs Predicted values - (N={len(df_2_intersection)})\nTarget={target}, Features=Anthropometrics and Age", fontsize=16, fontweight="bold")

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

# changing the fontsize of ticks
ax[0].set_yticks(np.arange(min(ax[0].get_yticks()), max(ax[0].get_yticks()), 5))
ax[1].set_yticks(np.arange(min(ax[1].get_yticks()), max(ax[1].get_yticks()), 5))

ax[0].tick_params(axis="both", labelsize=20)
ax[1].tick_params(axis="both", labelsize=20)

legend0 = ax[0].legend(title="Gender", loc="upper right")  # Add legend
legend1= ax[1].legend(title="Gender", loc="upper right")  # Add legend

# Modify individual legend labels
female_n = len(df_2_intersection[df_2_intersection["gender"]==0])
male_n = len(df_2_intersection[df_2_intersection["gender"]==1])

legend0.get_texts()[0].set_text(f"Female: N={female_n}")
legend0.get_texts()[1].set_text(f"Male: N={male_n}")
legend1.get_texts()[0].set_text(f"Female: N={female_n}")
legend1.get_texts()[1].set_text(f"Male: N={male_n}")

plt.show()
plt.savefig(f"{population}_{target}.png")
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


