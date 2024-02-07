import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score 

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
y = sys.argv[6]
x = sys.argv[7]
session = sys.argv[8]
###############################################################################
df = load_hgs_predicted_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
)

df_corr, df_pvalue = load_spearman_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
)

###############################################################################
df = df.rename(columns={f"1st_scan_{target}_predicted":"hgs_predicted", f"1st_scan_{target}_actual":"hgs_actual"})
df_corr = df_corr.rename(columns={f"1st_scan_{target}_predicted":"hgs_predicted", f"1st_scan_{target}_actual":"hgs_actual"})
df_corr = df_corr.rename(index={f"1st_scan_{target}_predicted":"hgs_predicted", f"1st_scan_{target}_actual":"hgs_actual"})
print("===== Done! =====")
embed(globals(), locals())
###############################################################################

custom_palette = {1: '#069AF3', 0: 'red'}

fig = plt.figure(figsize=(12,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

g = sns.jointplot(data=df, x="hgs_actual", y="hgs_predicted", hue="gender", palette=custom_palette, marker='H', joint_kws={'alpha': 0.3})

# , marginal_kws={'kde':True, 'common_norm':False})

for gender_type, gr in df.groupby(df['gender']):
    slope, intercept, r_value, p_value, std_err = linregress(gr["hgs_actual"], gr["hgs_predicted"])
    if gr['gender'].any() == 0:
        female_corr = pearsonr(gr["hgs_predicted"], gr["hgs_actual"])[0]
        print(female_corr)
    elif gr['gender'].any() == 1:
        male_corr = pearsonr(gr["hgs_predicted"], gr["hgs_actual"])[0]
        print(male_corr)
    p = sns.regplot(x="hgs_actual", y="hgs_predicted", data=gr, scatter=False, ax=g.ax_joint, color='darkgrey', line_kws={'label': f'{gender_type} Regression (r={r_value:.2f})'})
    print(r_value)
# for _,gr in df.groupby(df['gender']):
#     print(gr)
    
#     if gr['gender'].any() == 1:
#         custom_color = "#069AF3"
#     elif gr['gender'].any() == 0:
#         custom_color = "red"
#     print(custom_color)
#     sns.jointplot(x="hgs_actual", y="hgs_predicted", data=gr, kind="hex", color=custom_color, marginal_kws={'kde':True, 'common_norm':False})
#     # Access the hexbin plot and set alpha
#     hexbin = g.ax_joint.collections[0]
#     hexbin.set_alpha(0)
    
# remove the legend from ax_joint
g.ax_joint.legend_.remove()

g.fig.suptitle(f"{population} {mri_status}: {target}", fontsize=10, fontweight="bold")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

g.ax_joint.set_xlabel("True HGS", fontsize=12, fontweight="bold")
g.ax_joint.set_ylabel("Predicted HGS", fontsize=12, fontweight="bold")

xmin, xmax = g.ax_joint.get_xlim()
g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))

plt.show()
plt.savefig(f"jointplot_{population} {mri_status}: {target}.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())

###############################################################################
df1 = df[df['gender']==0]
df2 = df[df['gender']==1]
# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(df1['hgs_predicted'], df1['hgs_actual'], df1[f'1st_scan_{target}_(actual-predicted)'], cmap='r' )
ax.scatter(df2['hgs_predicted'], df2['hgs_actual'], df2[f'1st_scan_{target}_(actual-predicted)'], cmap='b')

plt.show()
plt.savefig(f"3d_{population} {mri_status}: {target}.png")

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
###############################################################################
df_combine = pd.DataFrame()
df_predicted = df[['gender', 'hgs_predicted']]
df_predicted.loc[:, 'hgs_type'] = 'hgs_predicted'
df_predicted = df_predicted.rename(columns={"hgs_predicted": "hgs"})
df_actual = df[['gender', 'hgs_actual']]
df_actual.loc[:, 'hgs_type'] = 'hgs_actual'
df_actual = df_actual.rename(columns={"hgs_actual": "hgs"})

df_combine = pd.concat([df_actual, df_predicted])
df_combine['gender'] = df_combine['gender'].replace({0: 'F', 1: 'M'})
df_combine.loc[:, 'gender_hgs_type'] = df_combine['gender'] + "-" + df_combine['hgs_type'].str.replace("hgs_", "")

df_combine = df_combine.sort_values(by=['gender_hgs_type'])

custom_palette = {"F-actual": 'red', 'F-predicted': 'red', "M-actual": '#069AF3', 'M-predicted':'#069AF3'}

fig = plt.figure(figsize=(6,6))
ax.set_aspect('equal')
ax = sns.violinplot(data=df_combine, x="gender_hgs_type", y="hgs", hue="gender_hgs_type", hue_order=['F-actual', 'F-predicted', 'M-actual', 'M-predicted'], palette=custom_palette, linewidth=1, dodge=True)
# Adjust transparency of the violins
for collection in ax.collections:
    collection.set_alpha(0.3)

plt.title(f"True vs predicted {target}", fontsize=15, fontweight="bold")

# plt.xlabel("", fontsize=40, fontweight="bold")
plt.ylabel("HGS(kg)", fontsize=25, fontweight="bold")

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='5', fontsize='5', bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig("test_violin.png")
plt.close()


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

g = sns.jointplot(data=df, x="hgs_actual", y="hgs_predicted", hue="gender", palette=custom_palette,  marker="$\circ$", s=50)

for gender_type, gr in df.groupby(df['gender']):
    slope, intercept, r_value, p_value, std_err = linregress(gr["hgs_actual"], gr["hgs_predicted"])
    if gr['gender'].any() == 0:
        female_corr = pearsonr(gr["hgs_predicted"], gr["hgs_actual"])[0]
        female_R2 = r2_score(gr["hgs_actual"], gr["hgs_predicted"])
        print(female_corr)
        print("female_r2=", female_R2)
    elif gr['gender'].any() == 1:
        male_corr = pearsonr(gr["hgs_predicted"], gr["hgs_actual"])[0]
        male_R2 = r2_score(gr["hgs_actual"], gr["hgs_predicted"])
        print(male_corr)
        print("male_r2=", male_R2)

    p = sns.regplot(x="hgs_actual", y="hgs_predicted", data=gr, scatter=False, ax=g.ax_joint, color='darkgrey', line_kws={'label': f'{gender_type} Regression (r={r_value:.2f})'})
    print(r_value)
    
# remove the legend from ax_joint
g.ax_joint.legend_.remove()

g.fig.suptitle(f"{population} {mri_status}: {target}", fontsize=10, fontweight="bold")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

g.ax_joint.set_xlabel("True HGS", fontsize=12, fontweight="bold")
g.ax_joint.set_ylabel("Predicted HGS", fontsize=12, fontweight="bold")

xmin, xmax = g.ax_joint.get_xlim()
ymin, ymax = g.ax_joint.get_ylim()
g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))

 # Plot regression line
g.ax_joint.plot([xmin, xmax], [ymin, ymax], 'k--')

plt.show()
plt.savefig(f"jointplot_circles_{population} {mri_status}: {target}{model_name}.png")
plt.close()

###############################################################################
###############################################################################
df_female = df[df['gender']==0]
df_male = df[df['gender']==1]

custom_palette = {1: '#069AF3', 0: 'red'}

# Calculate bias
bias = np.mean(df['hgs_actual'] - df['hgs_predicted'])

# Calculate standard deviation of differences
sd = np.std(df[f'1st_scan_{target}_(actual-predicted)'])

# Calculate limits of agreement
lower_limit = bias - 1.96 * sd
upper_limit = bias + 1.96 * sd

fig = plt.figure(figsize=(6,6))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

# Create the plot
sns.scatterplot(x=(df['hgs_actual'] + df['hgs_predicted']) / 2, y=df['hgs_actual'] - df['hgs_predicted'], hue='gender', data=df, palette=custom_palette,  marker="$\circ$", s=50)
# sns.scatter(x=(df_male['hgs_actual'] + df_male['hgs_predicted']) / 2), y=df_male['hgs_actual'] - df_male['hgs_predicted'], color='#069AF3', label='Male', edgecolors='#069AF3', facecolors='none')
# sns.scatter((df_female['hgs_actual'] + df_female['hgs_predicted']) / 2, df_female['hgs_actual'] - df_female['hgs_predicted'], color='red', label='Female', edgecolors='red', facecolors='none')

plt.axhline(y=bias, color='black', linestyle='--', linewidth=3, label='Bias')
plt.axhline(y=lower_limit, color='black', linestyle='-', linewidth=3, label='Lower Limit')
plt.axhline(y=upper_limit, color='black', linestyle='-', linewidth=3, label='Upper Limit')

# remove the legend from ax_joint
plt.legend().remove()

plt.title(f"Limits of Agreement Plot_{population} {mri_status}: {target}", fontsize=10, fontweight="bold")

plt.xlabel('HGS average (True and predicted)', fontsize=12, fontweight="bold")
plt.ylabel('HGS differences (True - Predicted)', fontsize=12, fontweight="bold")

# plt.tight_layout()
# Show the plot
plt.show()
plt.savefig(f"limit_agreement_plot_{population} {mri_status}: {target}.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
###############################################################################