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
target = sys.argv[5]
df_combined = pd.DataFrame()
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


# Define a custom palette with two blue colors
custom_palette = {0: '#800080', 1: '#000080'}
df_2_intersection["avg_2"] = (df_2_intersection[f"{target}_actual"] +  df_2_intersection[f"{target}_predicted"])/2
df_3_intersection["avg_3"] = (df_3_intersection[f"{target}_actual"] +  df_3_intersection[f"{target}_predicted"])/2

fig, ax = plt.subplots(2,1, figsize = (10,10))
sm.graphics.mean_diff_plot(df_2_intersection[f"{target}_actual"], df_2_intersection[f"{target}_predicted"], ax=ax[0])
ax[0].scatter(df_2_intersection["avg_2"], df_2_intersection[f"{target}_(actual-predicted)"], c=df_2_intersection.gender.map(custom_palette))
sm.graphics.mean_diff_plot(df_3_intersection[f"{target}_actual"], df_3_intersection[f"{target}_predicted"], ax=ax[1])
ax[1].scatter(df_3_intersection["avg_3"], df_3_intersection[f"{target}_(actual-predicted)"], c=df_3_intersection.gender.map(custom_palette))

ax[0].set_title("1st scan session")
ax[1].set_title("2nd scan ession")
#display Bland-Altman plot
plt.show()
plt.savefig(f"{population}_{target}_blandaltman_plot.png")


print("===== Done! =====")
embed(globals(), locals())