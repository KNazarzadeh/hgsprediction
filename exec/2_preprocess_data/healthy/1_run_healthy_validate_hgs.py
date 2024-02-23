
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
session = sys.argv[3]
data_set = sys.argv[4]

if mri_status == "nonmri":
    if data_set == "training":
        df = healthy_load_data.load_original_binned_train_data(population, mri_status)
    # elif data_set == "test":
    #     folder_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/splitted_data/nonmri/test_set/original_test_data"
    #     file_path = os.path.join(folder_path, "test_set_nonmri_healthy.csv")
    #     df = pd.read_csv(file_path, sep=',')
       
    #     df = df.drop(columns="index")
    #     df = df.rename(columns={"eid": "SubjectID"})
    #     df = df.set_index("SubjectID")
        
elif mri_status == "mri":
    df = healthy_load_data.load_original_data(population, mri_status)
data_processor = HealthyDataPreprocessor(df, mri_status, session)
# CHECK HGS AVAILABILITY
df = data_processor.check_hgs_availability(df)
# DATA VALIDATION
df = data_processor.validate_handgrips(df)

# Remove all columns with all NaN values
df = data_processor.remove_nan_columns(df)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]
print("===== Done! =====")
embed(globals(), locals())
if data_set == "test":
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        f"{mri_status}_{population}",
        "test_holdout_set",
        "validated_hgs_data",
        f"{session}_session_ukb"
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

        
    file_path = os.path.join(
        folder_path,
        f"female_validate_hgs_data.csv")
    df_female.to_csv(file_path, sep=',', index=True)
    
    file_path = os.path.join(
        folder_path,
        f"male_validate_hgs_data.csv")
    df_male.to_csv(file_path, sep=',', index=True)
    
    file_path = os.path.join(
        folder_path,
        f"both_gender_validate_hgs_data.csv")
    df.to_csv(file_path, sep=',', index=True)
    
else:
    healthy_save_data.save_validate_hgs_data(df_female, population, mri_status, session, "female")
    healthy_save_data.save_validate_hgs_data(df_male, population, mri_status, session, "male")
    healthy_save_data.save_validate_hgs_data(df, population, mri_status, session, "both_gender")


print("===== Done! =====")
embed(globals(), locals())