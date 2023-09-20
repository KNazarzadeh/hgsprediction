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
target = sys.argv[5]

stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_longitudinal = stroke.load_hgs_predicted_results(population, mri_status, session_column, model_name, feature_type, target, "both_gender")

if target == "hgs_L+R":
    target_string = target.replace("L", "Left").replace("R", "Right")
    target_string = target_string.replace("hgs_", "")
    target_string = "HGS(" + target_string + ")"
elif target in ["hgs_left", "hgs_right"]:
    target_string = target.replace("hgs_", "")
    target_string = target_string.capitalize()
    target_string = "HGS(" + target_string + ")"
    
# Create a figure and axis
fig, ax = plt.subplots(1,2, figsize=(16,10))

# Iterate through subjects and plot connecting lines
df_female = df_longitudinal[df_longitudinal["gender"]==0]
df_male = df_longitudinal[df_longitudinal["gender"]==1]
for idx, row in df_female.iterrows():
    x_values = [row["1st_pre-stroke_age"], row["1st_post-stroke_age"]]
    y_values = [row["1st_pre-stroke_hgs_L+R_actual"], row["1st_post-stroke_hgs_L+R_actual"]]
    ax[0].plot(x_values, y_values, marker='o', color="#800080")
for idx, row in df_male.iterrows():
    x_values = [row["1st_pre-stroke_age"], row["1st_post-stroke_age"]]
    y_values = [row["1st_pre-stroke_hgs_L+R_actual"], row["1st_post-stroke_hgs_L+R_actual"]]
    ax[0].plot(x_values, y_values, marker='o', color="#000080")
for idx, row in df_female.iterrows():
    x_values = [row["1st_pre-stroke_age"], row["1st_post-stroke_age"]]
    y_values = [row["1st_pre-stroke_hgs_L+R_actual"], row["1st_post-stroke_hgs_L+R_actual"]]
    ax[1].plot(x_values, y_values, marker='o', color="#800080")
for idx, row in df_male.iterrows():
    x_values = [row["1st_pre-stroke_age"], row["1st_post-stroke_age"]]
    y_values = [row["1st_pre-stroke_hgs_L+R_predicted"], row["1st_post-stroke_hgs_L+R_predicted"]]
    ax[1].plot(x_values, y_values, marker='o', color="#000080")

# Set labels and title
ax[0].set_xlabel('Age', fontsize=20, fontweight="bold")
ax[0].set_ylabel(f'Actual {target_string}', fontsize=20, fontweight="bold")
ax[1].set_xlabel('Age', fontsize=20, fontweight="bold")
ax[1].set_ylabel(f'Predicted {target_string}', fontsize=20, fontweight="bold")

ymin = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
ymax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])

ax[0].set_yticks(np.arange(10, 120, 5))
ax[1].set_yticks(np.arange(10, 120, 5))   

ax[0].tick_params(axis='y', labelsize=18)  # Adjust the font size (12 in this example)
ax[1].tick_params(axis='y', labelsize=18)  # Adjust the font size (12 in this example)

ax[0].tick_params(axis='x', labelsize=14)  # Adjust the font size (12 in this example)
ax[1].tick_params(axis='x', labelsize=14)  # Adjust the font size (12 in this example)

fig.suptitle(f'Changes in HGS and Age from Pre-Stroke to Post-Stroke for Each Subject\nTarget={target_string}', fontsize=20, fontweight="bold")

# Show the legend
ax[0].legend()
ax[1].legend()

# Show the plot
plt.show()
plt.savefig("hgs_age_predicted.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())