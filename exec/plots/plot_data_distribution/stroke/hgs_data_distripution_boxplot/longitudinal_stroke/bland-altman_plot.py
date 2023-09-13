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
    
selected_cols = [col for col in df_longitudinal.columns if any(item in col for item in ["actual", "predicted"])]
df_selected = df_longitudinal[selected_cols]

df_selected.insert(0, "gender", df_longitudinal["gender"])

# Define a custom palette with two blue colors
custom_palette = {0: '#800080', 1: '#000080'}
df_selected["avg_pre"] = (df_selected[f"1st_pre-stroke_{target}_actual"] +  df_selected[f"1st_pre-stroke_{target}_predicted"])/2
df_selected["avg_post"] = (df_selected[f"1st_post-stroke_{target}_actual"] +  df_selected[f"1st_post-stroke_{target}_predicted"])/2

fig, ax = plt.subplots(2,1, figsize = (10,10))
sm.graphics.mean_diff_plot(df_selected[f"1st_pre-stroke_{target}_actual"], df_selected[f"1st_pre-stroke_{target}_predicted"], ax=ax[0])
ax[0].scatter(df_selected["avg_pre"], df_selected[f"1st_pre-stroke_{target}_(actual-predicted)"], c=df_selected.gender.map(custom_palette))
sm.graphics.mean_diff_plot(df_selected[f"1st_post-stroke_{target}_actual"], df_selected[f"1st_post-stroke_{target}_predicted"], ax=ax[1])
ax[1].scatter(df_selected["avg_post"], df_selected[f"1st_post-stroke_{target}_(actual-predicted)"], c=df_selected.gender.map(custom_palette))

ax[0].set_title("pre-stroke")
ax[1].set_title("post-stroke")
#display Bland-Altman plot
plt.show()
plt.savefig(f"{population}_{target}_blandaltman_plot.png")


print("===== Done! =====")
embed(globals(), locals())