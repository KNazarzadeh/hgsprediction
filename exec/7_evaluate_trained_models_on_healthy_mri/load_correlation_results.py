
import os
import sys
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set = sys.argv[9]
correlation_type = sys.argv[10]
gender = sys.argv[11]
session = sys.argv[12]

###############################################################################
if confound_status == "0":
        confound = "without_confound_removal"
else:
    confound = "with_confound_removal"
###############################################################################    
if correlation_type == "pearson":
    correlation_func = pearsonr
elif correlation_type == "spearman":
    correlation_func = spearmanr
###############################################################################
folder_path = os.path.join(
                "/data",
                "project",
                "stroke_ukb",
                "knazarzadeh",
                "project_hgsprediction",  
                "results_hgsprediction",
                f"{population}",
                f"{mri_status}",
                f"{session}_session_ukb",
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",            
                "hgs_prediction_correlation_results",
            )
        
if(not os.path.isdir(folder_path)):
    os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_{correlation_type}_correlation_values.csv")
    
df_corr = pd.read_csv(file_path, sep=',', index_col=0)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_r2_values.csv")

df_r2_values = pd.read_csv(file_path, sep=',', index_col=0)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_MAE_values.csv")

df_mae_values = pd.read_csv(file_path, sep=',', index_col=0)
    
print("===== Done! =====")
embed(globals(), locals())