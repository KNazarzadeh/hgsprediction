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

###############################################################################
# Define residuals
residuals_pre = df_longitudinal[f"1st_pre-stroke_{target}_(actual-predicted)"]
residuals_post = df_longitudinal[f"1st_post-stroke_{target}_(actual-predicted)"]

predicted_pre = df_longitudinal[f"1st_pre-stroke_{target}_predicted"]
predicted_post = df_longitudinal[f"1st_post-stroke_{target}_predicted"]

# Calculate Spearman correlation coefficient
# spearman_corr, _ = spearmanr(actual_values, predicted_values)

custom_palette = sns.color_palette(['#800080', '#000080'])  # You can use any hex color codes you prefer

# Create a scatter plot of residuals vs. predicted values
fig, ax = plt.subplots(1,2, figsize=(12, 10))
ax0 = sns.scatterplot(data=df_longitudinal, x=predicted_pre, y=residuals_pre, hue="gender", palette=custom_palette, ax=ax[0])
ax1 = sns.scatterplot(data=df_longitudinal, x=predicted_post, y=residuals_post, hue="gender", palette=custom_palette, ax=ax[1])

ax0.axhline(y=0, color='red', linestyle='--', linewidth=1.5)  # Add a horizontal line at y=0 for reference
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)  # Add a horizontal line at y=0 for reference

ax0.set_title("Pre-stroke", fontsize=18, fontweight="bold")
ax1.set_title("Post-stroke", fontsize=18, fontweight="bold")

ax0.grid(True)
ax1.grid(True)

fig.suptitle(f"Residual plot -  (N={len(df_longitudinal)})\nTarget={target}, Features=Anthropometrics and Age", fontsize="14", fontweight="bold")

fig.text(0.5, 0.04, 'Predicted Values', ha='center', fontsize=20, fontweight="bold")
fig.text(0.04, 0.5, 'Residuals', va='center', rotation='vertical', fontsize=20, fontweight="bold")
ax0.set_xlabel("")
ax0.set_ylabel("")
ax1.set_xlabel("")
ax1.set_ylabel("")

xmin = min(ax0.get_xlim()[0], ax1.get_xlim()[0])
xmax = max(ax0.get_xlim()[1], ax1.get_xlim()[1])
ymin = min(ax0.get_ylim()[0], ax1.get_ylim()[0])
ymax = max(ax0.get_ylim()[1], ax1.get_ylim()[1])

ax0.set_xlim(xmin, xmax)
ax1.set_xlim(xmin, xmax)
ax0.set_ylim(ymin, ymax)
ax1.set_ylim(ymin, ymax)

legend0 = ax0.legend(title="Gender", loc="upper right")  # Add legend
legend1= ax1.legend(title="Gender", loc="upper right")  # Add legend

# Modify individual legend labels
female_n = len(df_longitudinal[df_longitudinal["gender"]==0])
male_n = len(df_longitudinal[df_longitudinal["gender"]==1])

legend0.get_texts()[0].set_text(f"Female: N={female_n}")
legend0.get_texts()[1].set_text(f"Male: N={male_n}")
legend1.get_texts()[0].set_text(f"Female: N={female_n}")
legend1.get_texts()[1].set_text(f"Male: N={male_n}")

plt.show()
plt.savefig(f"residual_{target}.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())
# Create a scatter plot of residuals vs. predicted values
fig, ax = plt.subplots(1,2, figsize=(12, 10))
ax0 = sns.histplot(data=df_longitudinal, x=predicted_pre, y=residuals_pre, hue="gender", palette=custom_palette, kde="True")
# ax1 = sns.scatterplot(data=df_longitudinal, x=predicted_post, y=residuals_post, hue="gender", palette=custom_palette, ax=ax[1])

ax0.axhline(y=0, color='red', linestyle='--', linewidth=1.5)  # Add a horizontal line at y=0 for reference
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)  # Add a horizontal line at y=0 for reference

ax0.set_title("Pre-stroke", fontsize=18, fontweight="bold")
ax1.set_title("Post-stroke", fontsize=18, fontweight="bold")

ax0.grid(True)
ax1.grid(True)

fig.suptitle(f"Residual plot -  (N={len(df_longitudinal)})\nTarget={target}, Features=Anthropometrics and Age", fontsize="14", fontweight="bold")

fig.text(0.5, 0.04, 'Predicted Values', ha='center', fontsize=20, fontweight="bold")
fig.text(0.04, 0.5, 'Residuals', va='center', rotation='vertical', fontsize=20, fontweight="bold")
ax0.set_xlabel("")
ax0.set_ylabel("")
ax1.set_xlabel("")
ax1.set_ylabel("")

xmin = min(ax0.get_xlim()[0], ax1.get_xlim()[0])
xmax = max(ax0.get_xlim()[1], ax1.get_xlim()[1])
ymin = min(ax0.get_ylim()[0], ax1.get_ylim()[0])
ymax = max(ax0.get_ylim()[1], ax1.get_ylim()[1])

ax0.set_xlim(xmin, xmax)
ax1.set_xlim(xmin, xmax)
ax0.set_ylim(ymin, ymax)
ax1.set_ylim(ymin, ymax)

legend0 = ax0.legend(title="Gender", loc="upper right")  # Add legend
legend1= ax1.legend(title="Gender", loc="upper right")  # Add legend

# Modify individual legend labels
female_n = len(df_longitudinal[df_longitudinal["gender"]==0])
male_n = len(df_longitudinal[df_longitudinal["gender"]==1])

legend0.get_texts()[0].set_text(f"Female: N={female_n}")
legend0.get_texts()[1].set_text(f"Male: N={male_n}")
legend1.get_texts()[0].set_text(f"Female: N={female_n}")
legend1.get_texts()[1].set_text(f"Male: N={male_n}")

plt.show()
plt.savefig(f"residual_{target}.png")
plt.close()