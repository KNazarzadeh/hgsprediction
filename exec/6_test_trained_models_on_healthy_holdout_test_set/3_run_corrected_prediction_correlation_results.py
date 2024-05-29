
import os
import sys
import pandas as pd
import numpy as np
from hgsprediction.load_results.healthy.load_corrected_prediction_results import load_corrected_prediction_results
from hgsprediction.save_results.healthy.save_corrected_prediction_correlation_results import save_corrected_prediction_correlation_results
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

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

# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
df = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################   
if correlation_type == "pearson":
    correlation_func = pearsonr
elif correlation_type == "spearman":
    correlation_func = spearmanr
###############################################################################         
true_hgs = df.loc[:, f"{target}"]
predicted_hgs = df.loc[:, f"{target}_predicted"]
delta_hgs = df.loc[:, f"{target}_delta(true-predicted)"]
corrected_predicted_hgs = df.loc[:, f"{target}_corrected_predicted"]
delta_corrected_hgs = df.loc[:, f"{target}_corrected_delta(true-predicted)"]

df_correlations = pd.DataFrame(index=["r_values", "p_values"])
df_r2_values = pd.DataFrame(index=["r2_values"])

df_correlations.loc["r_values", "true_vs_predicted"] = correlation_func(true_hgs, predicted_hgs)[0]
df_correlations.loc["r_values", "true_vs_delta"] = correlation_func(true_hgs, delta_hgs)[0]
df_correlations.loc["r_values", "true_vs_corrected_predicted"] = correlation_func(true_hgs, corrected_predicted_hgs)[0]
df_correlations.loc["r_values", "true_vs_corrected_delta"] = correlation_func(true_hgs, delta_corrected_hgs)[0]

df_correlations.loc["p_values", "true_vs_predicted"] = correlation_func(true_hgs, predicted_hgs)[1]
df_correlations.loc["p_values", "true_vs_delta"] = correlation_func(true_hgs, delta_hgs)[1]
df_correlations.loc["p_values", "true_vs_corrected_predicted"] = correlation_func(true_hgs, corrected_predicted_hgs)[1]
df_correlations.loc["p_values", "true_vs_corrected_delta"] = correlation_func(true_hgs, delta_corrected_hgs)[1]


df_r2_values.loc["r2_values", "true_vs_predicted"] = r2_score(true_hgs, predicted_hgs)
df_r2_values.loc["r2_values", "true_vs_delta"] = r2_score(true_hgs, delta_hgs)
df_r2_values.loc["r2_values", "true_vs_corrected_predicted"] = r2_score(true_hgs, corrected_predicted_hgs)
df_r2_values.loc["r2_values", "true_vs_corrected_delta"] = r2_score(true_hgs, delta_corrected_hgs)

save_corrected_prediction_correlation_results(
    df_correlations,
    df_r2_values,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds, 
    correlation_type,   
    data_set,    
)
print("===== Done! End =====")
embed(globals(), locals())