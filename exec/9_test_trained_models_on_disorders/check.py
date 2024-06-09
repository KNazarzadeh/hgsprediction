import math
import sys
import os
import numpy as np
import pandas as pd

from hgsprediction.load_results.disorder.load_disorder_matched_samples_results import load_disorder_matched_samples_results

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
confound_status = sys.argv[5]
n_repeats = sys.argv[6]
n_folds = sys.argv[7]
disorder_cohort = sys.argv[8]
visit_session = sys.argv[9]
n_samples = sys.argv[10]
target = sys.argv[11]
first_event = sys.argv[12]
##############################################################################
main_extracted_columns = ["gender", "handedness", "hgs_dominant", "hgs_dominant_side", "hgs_nondominant", "hgs_nondominant_side", 
                          "age", "bmi", "height", "waist_to_hip_ratio", 
                          "group", "time_point", "hgs_target", 
                          "true_hgs", "hgs_predicted", "hgs_delta", "hgs_corrected_predicted", "hgs_corrected_delta", 
                          "patient_id"]
##############################################################################
df_disorder = pd.DataFrame()
df_control = pd.DataFrame()
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
df_disorder_matched_female, df_mathced_controls_female = load_disorder_matched_samples_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "female",
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
    first_event,
)

df_disorder_matched_male, df_mathced_controls_male = load_disorder_matched_samples_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "male",
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
    first_event,    
)
print("===== Done! =====")
embed(globals(), locals())

