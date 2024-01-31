import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import seaborn as sns
from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import save_spearman_correlation_results
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
df_combined_models_scores = pd.DataFrame()

if target == "all":
    for tar in ["hgs_left", "hgs_right", "hgs_L+R"]:
        if model_name == "both_models":
            df_combined_targets = pd.DataFrame()
            for model in ["linear_svm", "random_forest"]:        
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
                    f"{tar}",
                    f"{model}",
                    "hgs_predicted_results",
                    )
                # Define the csv file path to save
                file_path = os.path.join(
                    folder_path,
                    f"both_gender_hgs_predicted_results.csv")
                print(file_path)
                
                df = pd.read_csv (file_path, sep=',', index_col=0)
                df.loc[:, 'model'] = model
                df.loc[:, 'target'] = tar      
                df = df.rename(columns={f"{tar}_(actual-predicted)":"hgs_(actual-predicted)"})
                df = df.rename(columns={f"{tar}":"hgs", f"{tar}_predicted":"hgs_predicted", f"{tar}_actual":"hgs_actual"})      
                df_combined_targets = pd.concat ([df_combined_targets, df], axis=0)
                
                
            df_combined_models_scores = pd.concat ([df_combined_models_scores, df_combined_targets], axis=0)

    
    df_combined_models_scores['model'] = df_combined_models_scores['model'].str.replace('_', ' ').str.capitalize()
    df_combined_models_scores['gender'] = df_combined_models_scores['gender'].replace({0: "Female", 1:"Male"})
    df_combined_models_scores['model_target'] = df_combined_models_scores['model'] + " " + df_combined_models_scores['target']

    print(df_combined_models_scores)

###############################################################################
def add_median_labels(ax, fmt='.3f'):
    xticks_positios_array = []
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=12)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df_combined_models_scores = df_combined_models_scores.sort_values(by="model", ascending=True)
custom_palette = {'Female': 'red', 'Male': '#069AF3'}

fig = plt.figure(figsize=(18,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 25,
                     "xtick.labelsize": 25})

ax = sns.set_style("whitegrid")
ax = sns.boxplot(data=df_combined_models_scores, x="model_target", y=f"hgs_(actual-predicted)", hue='gender', palette=custom_palette)   

xticks_positios_array = add_median_labels(ax)

# ax = sns.violinplot(data=df_combined_models_scores, x="model", y=f"{target}_(actual-predicted)", hue='gender',
#                palette=custom_palette, linewidth=3)

plt.title(f"Compare models for prediction error onHoldout test set - Anthropometrics+Age features - {target}", fontsize=20, fontweight="bold")

plt.xlabel("Model", fontsize=40, fontweight="bold")
plt.ylabel("Prediction error \n (True-Predicted) HGS", fontsize=40, fontweight="bold")

# xmin, xmax = ax.get_xlim()
# ymin, ymax = ax.get_ylim()
# ax.set_yticks(np.arange(ymin, round(ymax), .5))

new_xticklabels = ["Left HGS", "Right HGS", "(L+R) HGS", "Left HGS", "Right HGS", "(L+R) HGS"]  # Replace with your desired labels
ax.set_xticklabels(new_xticklabels, fontsize=18, weight='bold')

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig(f"gender_specified_both_model_{target}_violin.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())
###############################################################################

custom_palette = {'Female': 'red', 'Male': '#069AF3'}


fig = plt.figure(figsize=(12,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

g = sns.jointplot(data=df_combined_models_scores, x=f"{target}_actual", y=f"{target}_predicted", hue="gender", palette=custom_palette,  marker="$\circ$", s=50)

# , marginal_kws={'kde':True, 'common_norm':False})

for gender_type, gr in df.groupby(df['gender']):
    slope, intercept, r_value, p_value, std_err = linregress(gr[f"{target}_actual"], gr[f"{target}_predicted"])
    if gr['gender'].any() == 0:
        female_corr = pearsonr(gr[f"{target}_predicted"], gr[f"{target}_actual"])[0]
        print(female_corr)
    elif gr['gender'].any() == 1:
        male_corr = pearsonr(gr[f"{target}_predicted"], gr[f"{target}_actual"])[0]
        print(male_corr)
    p = sns.regplot(x=f"{target}_actual", y=f"{target}_predicted", data=gr, scatter=False, ax=g.ax_joint, color='darkgrey', line_kws={'label': f'{gender_type} Regression (r={r_value:.2f})'})
    print(r_value)

# remove the legend from ax_joint
g.ax_joint.legend_.remove()

g.fig.suptitle(f"{population} {mri_status}: {target}", fontsize=10, fontweight="bold")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

g.ax_joint.set_xlabel("True HGS", fontsize=12, fontweight="bold")
g.ax_joint.set_ylabel("Predicted HGS", fontsize=12, fontweight="bold")

xmin, xmax = g.ax_joint.get_xlim()
g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))

plt.show()
plt.savefig(f"jointplot_{target}.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())
