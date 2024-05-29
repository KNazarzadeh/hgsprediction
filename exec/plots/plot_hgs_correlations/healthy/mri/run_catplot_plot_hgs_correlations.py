import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results.disorder import save_spearman_correlation_results
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
model_name = sys.argv[4]
y = sys.argv[5]
x = sys.argv[6]
session = sys.argv[7]
###############################################################################
df_combine = pd.DataFrame()
for tar in ["hgs_left", "hgs_right", "hgs_L+R"]:
    df = load_hgs_predicted_results(
        population,
        mri_status,
        model_name,
        feature_type,
        tar,
        "both_gender",
        session,
    )
    df.loc[:, 'target'] = tar
    df = df.rename(columns={f"1st_scan_{tar}_predicted":"hgs_predicted", f"1st_scan_{tar}_actual":"hgs_actual", f"1st_scan_{tar}_(actual-predicted)":"actual-predicted"})

    df_combine = pd.concat([df_combine, df[['1st_scan_age', 'gender', 'hgs_actual', 'hgs_predicted', 'actual-predicted', 'target']]])

df_combine['gender'] = df_combine['gender'].replace({0: 'Female', 1: 'Male'})
print("===== Done! =====")
embed(globals(), locals())
#############################################################################
###############################################################################

fig = plt.figure(figsize=(15,15))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid")

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    data=df_combine, x="target", y="actual-predicted", col="gender",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)


# # remove the legend from ax_joint
# g.ax_joint.legend_.remove()

# g.fig.suptitle(f"{population} {mri_status}:compare targets", fontsize=10, fontweight="bold")
# g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

# g.ax_joint.set_xlabel("True HGS", fontsize=12, fontweight="bold")
g.set_ylabels("Prediction error")

# xmin, xmax = g.ax_joint.get_xlim()
# g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))

plt.show()
plt.savefig(f"catplot_{population} {mri_status}_compare_targets.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())

###############################################################################
#############################################################################
###############################################################################
