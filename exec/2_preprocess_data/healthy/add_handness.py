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
data_set = sys.argv[3]

if mri_status == "nonmri":
    if data_set == "training":
        df = healthy_load_data.load_original_binned_train_data(population, mri_status)
        df_nonmri = pd.read_csv("/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/nonmri_healthy/nonmri_healthy.csv", sep=',', low_memory=False)

        df_nonmri = df_nonmri.rename(columns={"eid": "SubjectID"})
        df_nonmri = df_nonmri.set_index("SubjectID")
        print("===== Done! =====")
        embed(globals(), locals())
        # Extract columns "1707-0.0", "1707-1.0", "1707-2.0" for handness
        handness = df.loc[:, ["1707-0.0", "1707-1.0", "1707-2.0"]]
        
        # Find indices with NaN in the first column of handness
        index_withNaN = handness[handness["1707-0.0"].isna()].index
        # Replace NaN in the first column with the max of the corresponding row
        handness.loc[index_withNaN, "1707-0.0"] = np.nanmax(handness.loc[index_withNaN, :], axis=1)
        
        # Find indices where the first column equals -3 and set them to NaN
        index_no_answer = handness.loc[:, "1707-0.0"] == -3
        handness.loc[index_no_answer, "1707-0.0"] = np.nan
        
        # Remove all columns except the first
        handness = handness.loc[:, "1707-0.0"]
        
        # Find indices for left-handed, right-handed, and other
        index_left = handness[:, 0] == 2
        index_right = handness[:, 0] == 1
        index_other = (handness[:, 0] != 1) & (handness[:, 0] != 2)
        print("===== Done! =====")
        embed(globals(), locals())
        # Initialize strength with NaNs
        strength = np.full(len(df), np.nan)
        
        
        # Assign strength values based on handness
        strength[index_left] = df_nonmri[index_left, '46-0.0']
        strength[index_right] = df_nonmri[index_right, '47-0.0']
        strength[index_other] = np.nanmax(df_nonmri[index_other, ['46-0.0', '47-0.0']], axis=1)

        
                
        # Find outliers where strength is less than 4 and set them to NaN
        index_outlier = strength < 4
        strength[index_outlier] = np.nan
        
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
        
    elif data_set == "test":
        
        df = healthy_load_data.load_original_nonmri_test_data(population, mri_status)
        print("===== Done! =====")
        embed(globals(), locals())
        df_nonmri = pd.read_csv("/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/nonmri_healthy/nonmri_healthy.csv", sep=',', low_memory=False)

        df_nonmri = df_nonmri.rename(columns={"eid": "SubjectID"})
        df_nonmri = df_nonmri.set_index("SubjectID")
        
        df_tmp = df_nonmri[df_nonmri.index.isin(df.index)]
        df_tmp = df_tmp.reindex(df.index)

        df = pd.concat([df, df_tmp.loc[:, ["1707-1.0", "1707-2.0"]]], axis=1)

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
    df.to_csv(file_path, sep=',', index=True)
    
print("===== Done! =====")
embed(globals(), locals())

