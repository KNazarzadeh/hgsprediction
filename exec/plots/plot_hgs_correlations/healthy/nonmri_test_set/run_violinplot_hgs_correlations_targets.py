import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.metrics import r2_score 


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

df = df.rename(columns={f"{target}_(actual-predicted)":"hgs_(actual-predicted)"})
df = df.rename(columns={f"{target}":"hgs", f"{target}_predicted":"hgs_predicted", f"{target}_actual":"hgs_actual"})
###############################################################################
###############################################################################

# Reshape the DataFrame for Seaborn
# Melt the DataFrame based on 'true' and 'predicted' hgs
melted_df = pd.melt(df, id_vars=['gender'], value_vars=['hgs_actual', 'hgs_predicted'],
                    var_name='hgs_type', value_name='hgs_values')
print("===== Done! =====")
embed(globals(), locals())
custom_palette = {1: 'darkblue', 0: 'red'}

fig = plt.figure(figsize=(12,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

sns.violinplot(x='hgs_type', y='hgs_values', hue='gender', data=melted_df, inner="quartile", palette=custom_palette)

# Add title and labels
plt.title('Violin Plot of HGS Values by HGS Type and Gender')
plt.xlabel('HGS Type')
plt.ylabel('HGS Values')

plt.show()
plt.savefig(f"violine_test_set_{model_name}_{target}.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())
