
import sys
import os
import pandas as pd
import numpy as np
from hgsprediction.load_results.healthy.load_corrected_prediction_results import load_corrected_prediction_results

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
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set = sys.argv[9]
gender = sys.argv[10]
###############################################################################   
df_corrected_hgs_session_0 = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    "0",
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)
###############################################################################
df_corrected_hgs_session_1 = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    "1",
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)

###############################################################################

df_session_0 = df_corrected_hgs_session_0[df_corrected_hgs_session_0.index.isin(df_corrected_hgs_session_1.index)]

df_session_1 = df_corrected_hgs_session_1[df_corrected_hgs_session_1.index.isin(df_corrected_hgs_session_0.index)]

###############################################################################
# Reorder the index of df1 to match the index order of df2
df_session_0 = df_session_0.reindex(df_session_1.index)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
for session in ["0", "1"]:
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "results_hgsprediction",
        "test_retest_reliability",
        f"{population}",
        f"{mri_status}",
        f"{feature_type}",
        f"{target}",
        f"longitudinal_session_ukb",
        f"{session}_session_ukb",
        "extracted_data_by_feature_and_target",
        )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
    folder_path,
    f"{gender}_extracted_data_by_feature_and_target.csv")

    # Save the dataframe to csv file path
    if session == "0":
        df_session_0.to_csv(file_path, sep=',', index="True")
    elif session == "1":
        # Save the dataframe to csv file path
        df_session_1.to_csv(file_path, sep=',', index="True")

print("===== Done! =====")
embed(globals(), locals())