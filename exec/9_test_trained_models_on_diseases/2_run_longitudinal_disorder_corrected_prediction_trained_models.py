import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.prediction_corrector_model import prediction_corrector_model
from hgsprediction.load_results.load_disorder_hgs_predicted_results import load_disorder_hgs_predicted_results
from hgsprediction.save_results.save_disorder_corrected_prediction_results import save_disorder_corrected_prediction_results
from hgsprediction.save_results.save_disorder_corrected_prediction_correlation_results import save_disorder_corrected_prediction_correlation_results

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

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
gender = sys.argv[11]

###############################################################################

slope, intercept = prediction_corrector_model(
    "healthy",
    "nonmri",
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
)

print("slope=", slope)
print("intercept=", intercept)


##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

df = load_disorder_hgs_predicted_results(
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
)

###############################################################################
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}"
        
    # Beheshti Method:
    df.loc[:, f"{prefix}_{target}_corrected_predicted"] = (df.loc[:, f"{prefix}_{target}_predicted"] + ((slope * df.loc[:, f"{prefix}_{target}"]) + intercept))
    df.loc[:, f"{prefix}_{target}_corrected_delta(true-predicted)"] = df.loc[:, f"{prefix}_{target}"] - df.loc[:, f"{prefix}_{target}_corrected_predicted"]

###############################################################################
save_disorder_corrected_prediction_results(
    df,
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
)
print(df)
print("===== Done! End =====")
embed(globals(), locals())
###############################################################################
df_correlations = pd.DataFrame()
df_p_values = pd.DataFrame()
df_r2_values = pd.DataFrame()

for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}"

    df_correlations.loc[0, f"{prefix}_r_values_true_predicted"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_predicted"])[0]
    df_correlations.loc[0, f"{prefix}_r_values_true_delta"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_delta(true-predicted)"])[0]
    df_correlations.loc[0, f"{prefix}_r_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_corrected_predicted"])[0]
    df_correlations.loc[0, f"{prefix}_r_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_corrected_delta(true-predicted)"])[0]

    df_p_values.loc[0, f"{prefix}_r_values_true_predicted"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_predicted"])[1]
    df_p_values.loc[0, f"{prefix}_r_values_true_delta"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_delta(true-predicted)"])[1]
    df_p_values.loc[0, f"{prefix}_r_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_corrected_predicted"])[1]
    df_p_values.loc[0, f"{prefix}_r_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_corrected_delta(true-predicted)"])[1]


    df_r2_values.loc[0, f"{prefix}_r2_values_true_predicted"] = r2_score(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_predicted"])
    df_r2_values.loc[0, f"{prefix}_r2_values_true_delta"] = r2_score(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_delta(true-predicted)"])
    df_r2_values.loc[0, f"{prefix}_r2_values_true_corrected_predicted"] = r2_score(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_corrected_predicted"])
    df_r2_values.loc[0, f"{prefix}_r2_values_true_corrected_delta"] = r2_score(df.loc[:, f"{prefix}_{target}"],df.loc[:, f"{prefix}_{target}_corrected_delta(true-predicted)"])


save_disorder_corrected_prediction_correlation_results(
    df_correlations,
    df_p_values,
    df_r2_values,
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
)
print("===== Done! End =====")
embed(globals(), locals())