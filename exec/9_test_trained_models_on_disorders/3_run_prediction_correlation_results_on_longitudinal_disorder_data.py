import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results.disorder import load_disorder_corrected_prediction_results
from hgsprediction.save_results.disorder.save_disorder_prediction_correlation_results import save_disorder_prediction_correlation_results
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
disorder_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
model_name = sys.argv[7]
confound_status = sys.argv[8]
n_repeats = sys.argv[9]
n_folds = sys.argv[10]
correlation_type = sys.argv[11]
gender = sys.argv[12]
first_event = sys.argv[13]
###############################################################################
if correlation_type == "pearson":
    correlation_func = pearsonr
elif correlation_type == "spearman":
    correlation_func = spearmanr
##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
###############################################################################
df = load_disorder_corrected_prediction_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
    first_event,
)
print(df)
###############################################################################
df_correlations = pd.DataFrame(index=["r_values", "p_values"])
df_r2_values = pd.DataFrame(index=["r2_values"])
df_mae_values = pd.DataFrame(index=["MAE_values"])

for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}"
    #----------------------------------------------------#  
    true_hgs = df.loc[:, f"{prefix}_{target}"]
    predicted_hgs = df.loc[:, f"{prefix}_{target}_predicted"]
    delta_hgs = df.loc[:, f"{prefix}_{target}_delta(true-predicted)"]
    corrected_predicted_hgs = df.loc[:, f"{prefix}_{target}_corrected_predicted"]
    delta_corrected_hgs = df.loc[:, f"{prefix}_{target}_corrected_delta(true-predicted)"]
    #----------------------------------------------------#  
    df_correlations.loc["r_values", f"{prefix}_true_vs_predicted"] = correlation_func(true_hgs, predicted_hgs)[0]
    df_correlations.loc["r_values", f"{prefix}_true_vs_delta"] = correlation_func(true_hgs, delta_hgs)[0]
    df_correlations.loc["r_values", f"{prefix}_true_vs_corrected_predicted"] = correlation_func(true_hgs, corrected_predicted_hgs)[0]
    df_correlations.loc["r_values", f"{prefix}_true_vs_corrected_delta"] = correlation_func(true_hgs, delta_corrected_hgs)[0]

    df_correlations.loc["p_values", f"{prefix}_true_vs_predicted"] = correlation_func(true_hgs, predicted_hgs)[1]
    df_correlations.loc["p_values", f"{prefix}_true_vs_delta"] = correlation_func(true_hgs, delta_hgs)[1]
    df_correlations.loc["p_values", f"{prefix}_true_vs_corrected_predicted"] = correlation_func(true_hgs, corrected_predicted_hgs)[1]
    df_correlations.loc["p_values", f"{prefix}_true_vs_corrected_delta"] = correlation_func(true_hgs, delta_corrected_hgs)[1]


    df_r2_values.loc["r2_values", f"{prefix}_true_vs_predicted"] = r2_score(true_hgs, predicted_hgs)
    df_r2_values.loc["r2_values", f"{prefix}_true_vs_delta"] = r2_score(true_hgs, delta_hgs)
    df_r2_values.loc["r2_values", f"{prefix}_true_vs_corrected_predicted"] = r2_score(true_hgs, corrected_predicted_hgs)
    df_r2_values.loc["r2_values", f"{prefix}_true_vs_corrected_delta"] = r2_score(true_hgs, delta_corrected_hgs)

    df_mae_values.loc["MAE_values", f"{prefix}_true_vs_predicted"] = mean_absolute_error(true_hgs, predicted_hgs)
    df_mae_values.loc["MAE_values", f"{prefix}_true_vs_delta"] = mean_absolute_error(true_hgs, delta_hgs)
    df_mae_values.loc["MAE_values", f"{prefix}_true_vs_corrected_predicted"] = mean_absolute_error(true_hgs, corrected_predicted_hgs)
    df_mae_values.loc["MAE_values", f"{prefix}_true_vs_corrected_delta"] = mean_absolute_error(true_hgs, delta_corrected_hgs)

save_disorder_prediction_correlation_results(
    df_correlations,
    df_r2_values,
    df_mae_values,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
    first_event,
    correlation_type, 
)

print("===== Done! =====")
embed(globals(), locals())