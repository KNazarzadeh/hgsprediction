import sys
import numpy as np
import pandas as pd

from hgsprediction.load_results.disorder import load_disorder_matched_samples_results
from hgsprediction.save_results.disorder import save_disorder_comparison_matched_samples_disorder_vs_healthy_result
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
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
disorder_cohort = sys.argv[9]
visit_session = sys.argv[10]
gender = sys.argv[11]
n_samples = sys.argv[12]
first_event = sys.argv[13]
###############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
###############################################################################
df_disorder, df_control = load_disorder_matched_samples_results(
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
    n_samples,
    first_event,
)
print("===== Done! End =====")
embed(globals(), locals())
###############################################################################
# Initialize an empty DataFrame
df = pd.DataFrame()

# Loop through the episodes and add columns to the DataFrame
for episode in ["pre", "post"]:
    column_names = [f"{population}_{episode}_time_point", f"controls_{episode}_time_point", f"difference_{episode}_time_point"]
    for name in column_names:
        df.insert(len(df.columns), name, None)
    
    for metric in ["age", "bmi", "height", "waist_to_hip_ratio", target]:
        mean_disorder = df_disorder[f"1st_{episode}-{population}_{metric}"].mean()
        mean_control = df_control[f"1st_{episode}-{population}_{metric}"].mean()
        difference = mean_disorder - mean_control
        
        df.loc[f"{metric}_mean", f"{population}_{episode}_time_point"] = mean_disorder
        df.loc[f"{metric}_mean", f"controls_{episode}_time_point"] = mean_control
        df.loc[f"{metric}_mean", f"difference_{episode}_time_point"] = difference

print(df)
###############################################################################
save_disorder_comparison_matched_samples_disorder_vs_healthy_result(
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
    n_samples,
    first_event,
)

print("===== Done! End =====")
embed(globals(), locals())