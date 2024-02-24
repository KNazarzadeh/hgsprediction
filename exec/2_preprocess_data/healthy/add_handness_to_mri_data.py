import sys
import os
import pandas as pd
import numpy as np
from hgsprediction.load_data import healthy_load_data
from hgsprediction.data_preprocessing import HealthyDataPreprocessor
from hgsprediction.save_data import healthy_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

if mri_status == "mri":
    df = healthy_load_data.load_original_data(population, mri_status)

handness = df[["1707-0.0", "1707-1.0", "1707-2.0"]]
# Replace NaN values in the first column with maximum of respective row
index_NaN = handness[handness.loc[:, "1707-0.0"].isna()].index
max_value = handness.loc[index_NaN, ["1707-1.0", "1707-2.0"]].max(axis=1)
handness.loc[index_NaN, "1707-0.0"] = max_value

# Replace occurrences of -3 in the first column with NaN
index_no_answer = handness[handness.loc[:, "1707-0.0"] == -3.0].index
handness.loc[index_no_answer, "1707-0.0"] = np.NaN

# Remove all columns except the first one
df.loc[:, "handness"] = handness.loc[:, "1707-0.0"]


folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "original_data",
        f"{mri_status}_{population}",
    )
    
# Define the csv file path to load
file_path = os.path.join(
    folder_path,
    f"{mri_status}_{population}.csv")

# Load the dataframe from csv file path
df.to_csv(file_path, sep=',', index=True)

print("===== Done! =====")
embed(globals(), locals())
