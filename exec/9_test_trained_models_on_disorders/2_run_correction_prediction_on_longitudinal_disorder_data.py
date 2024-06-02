import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.correction_predicted_hgs import prediction_corrector_model
from hgsprediction.load_results.disorder.load_disorder_hgs_predicted_results import load_disorder_hgs_predicted_results
from hgsprediction.save_results.disorder.save_disorder_corrected_prediction_results import save_disorder_corrected_prediction_results

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
first_event = sys.argv[12]
###############################################################################
slope, intercept = prediction_corrector_model(
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
)

print(slope)
print(intercept)
##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
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
    first_event,
)
###############################################################################
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}"
    #----------------------------------------------------#
    # Beheshti Method:
    true_hgs = df.loc[:, f"{prefix}_{target}"]
    predicted_hgs = df.loc[:, f"{prefix}_{target}_predicted"]

    # Beheshti Method:
    df.loc[:, f"{prefix}_{target}_corrected_predicted"] = (predicted_hgs + ((slope * true_hgs) + intercept))
    df.loc[:, f"{prefix}_{target}_corrected_delta(true-predicted)"] = true_hgs - df.loc[:, f"{prefix}_{target}_corrected_predicted"]

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
    first_event,
)
print(df)
print("===== Done! End =====")
embed(globals(), locals())
