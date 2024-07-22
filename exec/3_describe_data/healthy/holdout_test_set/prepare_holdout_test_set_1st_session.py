import pandas as pd
import numpy as np
import sys
import os

from hgsprediction.load_data.healthy import load_healthy_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
session = sys.argv[5]
data_set = sys.argv[6]
###############################################################################
# load data # Extract data based on main features, extra features, target for each session and mri status:
if data_set == "holdout_test_set":
    df_original = load_healthy_data.load_original_nonmri_test_data(population, mri_status)
###############################################################################

nonmri_file_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/nonmri_healthy/nonmri_healthy.csv"

df_nonmri = pd.read_csv(nonmri_file_path, sep=',', low_memory=False)

df_nonmri = df_nonmri.rename(columns={"eid":"SubjectID"})
df_nonmri = df_nonmri.set_index("SubjectID")

print("===== Done! =====")
embed(globals(), locals())
###############################################################################

df_with_all_sessions = df_nonmri[df_nonmri.index.isin(df_original.index)]

###############################################################################


folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "splitted_data",
        f"{mri_status}",
        "test_set",
        "original_test_data",
    )
# Define the csv file path to load
file_path = os.path.join(
    folder_path,
    f"test_set_{mri_status}_{population}.csv")

# Load the dataframe from csv file path
df_with_all_sessions.to_csv(file_path, sep=',', index=True)
