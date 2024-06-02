
import os
import sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from hgsprediction.load_results.healthy import load_trained_model_results
from hgsprediction.correction_predicted_hgs import beheshti_correction_method

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
data_set = sys.argv[10]
correlation_type = sys.argv[11]
gender = sys.argv[12]
###############################################################################   
if correlation_type == "pearson":
    correlation_func = pearsonr
elif correlation_type == "spearman":
    correlation_func = spearmanr
###############################################################################
df = load_trained_model_results.load_prediction_hgs_on_validation_set(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    session,
    data_set,
)

###############################################################################
df_corrected = pd.DataFrame()

df_correlations = pd.DataFrame(index=["r_values", "p_values"])
df_r2_values = pd.DataFrame(index=["r2_values"])
df_mae_values = pd.DataFrame(index=["MAE_values"])

###############################################################################
model = LinearRegression()
###############################################################################
for fold in range(int(n_folds)):
    df_tmp = df[df['cv_fold']==fold]
    df_half = df_tmp.sample(frac=0.5, random_state=47)
    #-----------------------------------------------------------#
    # Beheshti Method:
    X = df_half.loc[:, f"{target}"].values.reshape(-1, 1)
    # y = df.loc[:, f"{target}_delta(true-predicted)"].values
    y = df_half.loc[:, f"{target}_delta(true-predicted)"].values

    model.fit(X, y)
    #-----------------------------------------------------------#
    slope = model.coef_[0]
    intercept = model.intercept_   
    #-----------------------------------------------------------#
    df_half_rest = df_tmp[~df_tmp.index.isin(df_half.index)]
    #-----------------------------------------------------------#
    # Beheshti Method:
    true_hgs = df_half_rest.loc[:, f"{target}"]
    predicted_hgs = df_half_rest.loc[:, f"{target}_predicted"]
    df_corrected_hgs = beheshti_correction_method(df_half_rest.copy(), target, true_hgs, predicted_hgs, slope, intercept)
    #-----------------------------------------------------------#
    true_hgs = df_corrected_hgs.loc[:, f"{target}"]
    predicted_hgs = df_corrected_hgs.loc[:, f"{target}_predicted"]
    delta_hgs = df_corrected_hgs.loc[:, f"{target}_delta(true-predicted)"]
    corrected_predicted_hgs = df_corrected_hgs.loc[:, f"{target}_corrected_predicted"]
    delta_corrected_hgs = df_corrected_hgs.loc[:, f"{target}_corrected_delta(true-predicted)"]
    #-----------------------------------------------------------#
    df_tmp_correlations = pd.DataFrame(index=["r_values", "p_values"])
    df_tmp_r2_values = pd.DataFrame(index=["r2_values"])
    df_tmp_mae_values = pd.DataFrame(index=["MAE_values"])
    
    df_tmp_correlations.loc["r_values", "cv_fold"] = fold
    df_tmp_correlations.loc["r_values", "true_vs_predicted"] = correlation_func(true_hgs, predicted_hgs)[0]
    df_tmp_correlations.loc["r_values", "true_vs_predicted"] = correlation_func(true_hgs, predicted_hgs)[0]
    df_tmp_correlations.loc["r_values", "true_vs_delta"] = correlation_func(true_hgs, delta_hgs)[0]
    df_tmp_correlations.loc["r_values", "true_vs_corrected_predicted"] = correlation_func(true_hgs, corrected_predicted_hgs)[0]
    df_tmp_correlations.loc["r_values", "true_vs_corrected_delta"] = correlation_func(true_hgs, delta_corrected_hgs)[0]
    
    df_tmp_correlations.loc["p_values", "cv_fold"] = fold
    df_tmp_correlations.loc["p_values", "true_vs_predicted"] = correlation_func(true_hgs, predicted_hgs)[1]
    df_tmp_correlations.loc["p_values", "true_vs_delta"] = correlation_func(true_hgs, delta_hgs)[1]
    df_tmp_correlations.loc["p_values", "true_vs_corrected_predicted"] = correlation_func(true_hgs, corrected_predicted_hgs)[1]
    df_tmp_correlations.loc["p_values", "true_vs_corrected_delta"] = correlation_func(true_hgs, delta_corrected_hgs)[1]

    df_tmp_r2_values.loc["r2_values", "cv_fold"] = fold
    df_tmp_r2_values.loc["r2_values", "true_vs_predicted"] = r2_score(true_hgs, predicted_hgs)
    df_tmp_r2_values.loc["r2_values", "true_vs_delta"] = r2_score(true_hgs, delta_hgs)
    df_tmp_r2_values.loc["r2_values", "true_vs_corrected_predicted"] = r2_score(true_hgs, corrected_predicted_hgs)
    df_tmp_r2_values.loc["r2_values", "true_vs_corrected_delta"] = r2_score(true_hgs, delta_corrected_hgs)

    df_tmp_mae_values.loc["MAE_values", "cv_fold"] = fold
    df_tmp_mae_values.loc["MAE_values", "true_vs_predicted"] = mean_absolute_error(true_hgs, predicted_hgs)
    df_tmp_mae_values.loc["MAE_values", "true_vs_delta"] = mean_absolute_error(true_hgs, delta_hgs)
    df_tmp_mae_values.loc["MAE_values", "true_vs_corrected_predicted"] = mean_absolute_error(true_hgs, corrected_predicted_hgs)
    df_tmp_mae_values.loc["MAE_values", "true_vs_corrected_delta"] = mean_absolute_error(true_hgs, delta_corrected_hgs)
    
    df_corrected = pd.concat([df_corrected, df_corrected_hgs], axis=0)

    if fold == 0:
        df_correlations = df_tmp_correlations.copy()
        df_r2_values = df_tmp_r2_values.copy()
        df_mae_values = df_tmp_mae_values.copy()
    else:
        df_correlations = pd.concat([df_correlations, df_tmp_correlations], axis=0)
        df_r2_values = pd.concat([df_r2_values, df_tmp_r2_values], axis=0)
        df_mae_values = pd.concat([df_mae_values, df_tmp_mae_values], axis=0)
    
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
if confound_status == "0":
        confound = "without_confound_removal"
else:
    confound = "with_confound_removal"
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
                f"{data_set}",
                f"{session}_session_ukb",
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",
                "hald_of_each_fold_training_rest_half_testing",
                "hgs_corrected_prediction_results",
            )

if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_corrected_prediction_data.csv")

df_corrected.to_csv(file_path, sep=',', index=True)
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
                f"{data_set}",
                f"{session}_session_ukb",
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",
                "hald_of_each_fold_training_rest_half_testing",
                "hgs_prediction_correlation_results",
            )

if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_{correlation_type}_correlation_values.csv")

df_correlations.to_csv(file_path, sep=',', index=True)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_r2_values.csv")

df_r2_values.to_csv(file_path, sep=',', index=True)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_MAE_values.csv")

df_mae_values.to_csv(file_path, sep=',', index=True)

print("===== Done! =====")
embed(globals(), locals())