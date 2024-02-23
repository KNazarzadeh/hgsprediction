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
    df_mri = pd.read_csv("/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/mri_healthy/mri_healthy.csv", sep=',', low_memory=False)

    df_mri = df_mri.rename(columns={"eid": "SubjectID"})
    df_mri = df_mri.set_index("SubjectID")

df_handness = df_mri[["1707-0.0", "1707-1.0", "1707-2.0"]]

df_handness_tmp = df_handness[df_handness.index.isin(df.index)]

df_handness_tmp = df_handness_tmp.reindex(df.index)

df_tmp = pd.concat([df, df_handness_tmp.loc[:, ["1707-1.0", "1707-2.0"]]], axis=1)

handness = df_tmp[["1707-0.0", "1707-1.0", "1707-2.0"]]
# Replace NaN values in the first column with maximum of respective row
index_NaN = handness[handness.loc[:, "1707-0.0"].isna()].index
max_value = handness.loc[index_NaN, ["1707-1.0", "1707-2.0"]].max(axis=1)
handness.loc[index_NaN, "1707-0.0"] = max_value

# Replace occurrences of -3 in the first column with NaN
index_no_answer = handness[handness.loc[:, "1707-0.0"] == -3.0].index
handness.loc[index_no_answer, "1707-0.0"] = np.NaN

# Remove all columns except the first one
df_tmp.loc[:, "handness"] = handness.loc[:, "1707-0.0"]

print("===== Done! =====")
embed(globals(), locals())

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
        "train_set",
        "original_binned_train_data"
    )
# Define the csv file path to load
file_path = os.path.join(
    folder_path,
    f"original_binned_train_data.csv")

df_tmp.to_csv(file_path, sep=',', index=True)