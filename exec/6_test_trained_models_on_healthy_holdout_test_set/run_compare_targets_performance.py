import os
import pandas as pd
import numpy as np
import sys


from hgsprediction.load_results import load_trained_models
from hgsprediction.load_data import load_healthy_data
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_pearson_hgs_correlation
from hgsprediction.save_results.healthy import save_correlation_results, \
                                               save_hgs_predicted_results
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
session = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
gender = sys.argv[9]
###############################################################################
# Assuming that you have already trained and instantiated the model as `model`

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
        "hgs_L+R",
        f"{model_name}",
        "hgs_predicted_results",
    )
    
if(not os.path.isdir(folder_path)):
    os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_predicted_results.csv")

df_l_plus_r = pd.read_csv(file_path, sep=',', index_col=0)


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
        "hgs_LI",
        f"{model_name}",
        "hgs_predicted_results",
    )
    
if(not os.path.isdir(folder_path)):
    os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_predicted_results.csv")

df_LI = pd.read_csv(file_path, sep=',', index_col=0)

print("===== Done! =====")
embed(globals(), locals())
# Model 1 Metrics
accuracy1 = accuracy_score(df_l_plus_r['hgs_L+R'], df_l_plus_r['hgs_L+R_predicted'])
precision1 = precision_score(df_l_plus_r['hgs_L+R'], df_l_plus_r['hgs_L+R_predicted'])
recall1 = recall_score(df_l_plus_r['hgs_L+R'], df_l_plus_r['hgs_L+R_predicted'])
f1_score1 = f1_score(df_l_plus_r['hgs_L+R'], df_l_plus_r['hgs_L+R_predicted'])
conf_matrix1 = confusion_matrix(df_l_plus_r['hgs_L+R'], df_l_plus_r['hgs_L+R_predicted'])

# Model 2 Metrics
accuracy1 = accuracy_score(df_LI['hgs_LI'], df_LI['hgs_LI_predicted'])
precision1 = precision_score(df_LI['hgs_LI'], df_LI['hgs_LI_predicted'])
recall1 = recall_score(df_LI['hgs_LI'], df_LI['hgs_LI_predicted'])
f1_score1 = f1_score(df_LI['hgs_LI'], df_LI['hgs_LI_predicted'])
conf_matrix1 = confusion_matrix(df_LI['hgs_LI'], df_LI['hgs_LI_predicted'])


print("===== Done! =====")
embed(globals(), locals())