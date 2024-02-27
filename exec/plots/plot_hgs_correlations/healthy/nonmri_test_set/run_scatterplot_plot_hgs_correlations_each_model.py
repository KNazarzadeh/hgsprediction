import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.metrics import r2_score 


import seaborn as sns
from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_pearson_hgs_correlation
from hgsprediction.save_results import save_correlation_results
from hgsprediction.load_results.healthy import load_hgs_predicted_results
from hgsprediction.load_results.healthy import load_spearman_correlation_results
from hgsprediction.save_plot.save_correlations_plot import healthy_save_correlations_plot
from hgsprediction.plots.plot_correlations import healthy_plot_hgs_correlations
from scipy.stats import linregress
from scipy.stats import pearsonr
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
###############################################################################
###############################################################################
folder_path = os.path.join(
"/data",
"project",
"stroke_ukb",
"knazarzadeh",
"project_hgsprediction",  
"results_hgsprediction",
f"{population}",
"nonmri_test_holdout_set",
f"{feature_type}",
f"{target}",
f"{model_name}",
"hgs_predicted_results",
)
# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"both_gender_hgs_predicted_results.csv")
print(file_path)

df = pd.read_csv (file_path, sep=',', index_col=0)

df = df.rename(columns={f"{target}_(true-predicted)":"hgs_(true-predicted)"})
df = df.rename(columns={f"{target}":"hgs", f"{target}_predicted":"hgs_predicted", f"{target}_true":"hgs_true"}) 
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
###############################################################################
custom_palette = {1: '#069AF3', 0: 'red'}

fig = plt.figure(figsize=(12,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

g = sns.jointplot(data=df, x="hgs_true", y="hgs_predicted", hue="gender", palette=custom_palette,  marker="$\circ$", s=50)

for gender_type, gr in df.groupby(df['gender']):
    slope, intercept, r_value, p_value, std_err = linregress(gr["hgs_true"], gr["hgs_predicted"])
    if gr['gender'].any() == 0:
        female_corr = pearsonr(gr["hgs_predicted"], gr["hgs_true"])[0]
        female_R2 = r2_score(gr["hgs_true"], gr["hgs_predicted"])
        print(female_corr)
        print("female_r2=", female_R2)
    elif gr['gender'].any() == 1:
        male_corr = pearsonr(gr["hgs_predicted"], gr["hgs_true"])[0]
        male_R2 = r2_score(gr["hgs_true"], gr["hgs_predicted"])
        print(male_corr)
        print("male_r2=", male_R2)
    color = custom_palette[gender_type]
    p = sns.regplot(x="hgs_true", y="hgs_predicted", data=gr, scatter=False, ax=g.ax_joint, color=color, line_kws={'label': f'{gender_type} Regression (r={r_value:.2f})'})
    print(r_value)
    
# remove the legend from ax_joint
g.ax_joint.legend_.remove()

g.fig.suptitle(f"{population}_test_set_{target}_{model_name}", fontsize=10, fontweight="bold")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

g.ax_joint.set_xlabel("True HGS", fontsize=12, fontweight="bold")
g.ax_joint.set_ylabel("Predicted HGS", fontsize=12, fontweight="bold")

xmin, xmax = g.ax_joint.get_xlim()
ymin, ymax = g.ax_joint.get_ylim()
g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))

 # Plot regression line
g.ax_joint.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

plt.show()
plt.savefig(f"new_scatter_test_set_{model_name}_{target}.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())
