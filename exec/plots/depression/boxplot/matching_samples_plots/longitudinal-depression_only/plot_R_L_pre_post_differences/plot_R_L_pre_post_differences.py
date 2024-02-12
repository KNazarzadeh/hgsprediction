import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import depression_load_data
from hgsprediction.load_results import depression

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

depression_cohort = "longitudinal-depression"
session_column = f"1st_{depression_cohort}_session"

longitudinal_mri = f"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/{population}/preprocessed_data/mri_{population}/longitudinal-{population}_data/1st_longitudinal-{population}_session_data/preprocessed_data/1st_longitudinal-{population}_session_preprocessed_data.csv"
longitudinal_nonmri = f"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/{population}/preprocessed_data/nonmri_{population}/longitudinal-{population}_data/1st_longitudinal-{population}_session_data/preprocessed_data/1st_longitudinal-{population}_session_preprocessed_data.csv"

df_longitudinal_mri = pd.read_csv(longitudinal_mri, sep=',', index_col=0)
df_longitudinal_nonmri = pd.read_csv(longitudinal_nonmri, sep=',', index_col=0)

df_long = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri])
df_longitudinal = depression.load_hgs_predicted_results(population, mri_status, session_column, model_name, feature_type, target, "both_gender")

df_long = df_long[df_long.index.isin(df_longitudinal.index)]

df_selected = pd.concat([df_long, df_longitudinal], axis=1)


df_selected["pre-post_hgs_right"] = df_selected[f"1st_pre-{population}_hgs_right"] - df_selected[f"1st_post-{population}_hgs_right"]
df_selected["pre-post_hgs_left"] = df_selected[f"1st_pre-{population}_hgs_left"] - df_selected[f"1st_post-{population}_hgs_left"]

df_selected['gender'] = df_selected['gender'].replace({0: 'female', 1: 'male'})

custom_palette = {'female': 'red', 'male': '#069AF3'}  # You can use any hex color codes you prefer

fig = plt.figure(figsize=(10,10))  # Adjust the figure size if needed
plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

ax = sns.scatterplot(x="pre-post_hgs_left", y="pre-post_hgs_right", data=df_selected, hue ="gender", palette=custom_palette, marker="$\circ$", s=120)
sns.regplot(x="pre-post_hgs_left", y="pre-post_hgs_right", data=df_selected, scatter=False, color='darkgrey')

# Set equal aspect ratio for the plot to make it square
ax.set_aspect('equal')

# Add labels and title
plt.xlabel("Left HGS", fontsize=12, fontweight="bold")
plt.ylabel("Right HGS", fontsize=12, fontweight="bold")

plt.title(f"Pre-Post HGS differences Right vs Left-{population.capitalize()}(N={len(df_selected)})", fontsize=15, fontweight="bold")

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='20', fontsize='18', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()


plt.show()
plt.savefig(f"scatter_pre-post_hgs_difference_{population}.png")
plt.close()

###############################################################################
###############################################################################



print("===== Done! =====")
embed(globals(), locals())