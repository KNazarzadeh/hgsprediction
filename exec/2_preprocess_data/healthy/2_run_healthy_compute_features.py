import sys
import os
import pandas as pd
from hgsprediction.load_data import healthy_load_data
from hgsprediction.compute_features import healthy_compute_features
from hgsprediction.save_data import healthy_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
# feature_type = sys.argv[3]
session = sys.argv[3]
data_set = sys.argv[4]

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
    file_path = os.path.join(
        folder_path,
        f"both_gender_validate_hgs_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
else:
    df = healthy_load_data.load_validate_hgs_data(population, mri_status, session, "both_gender")

# df = healthy_compute_features.compute_features(df, mri_status, feature_type, session)
df = healthy_compute_features.compute_features(df, mri_status, session)

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
        "preprocessed_data",
        f"{session}_session_ukb",
    )

    if(not os.path.isdir(folder_path)):
            os.makedirs(folder_path)
        
    file_path = os.path.join(
        folder_path,
        f"female_preprocessed_data_data.csv")
    df_female.to_csv(file_path, sep=',', index=True)
    
    file_path = os.path.join(
        folder_path,
        f"male_preprocessed_data_data.csv")
    df_male.to_csv(file_path, sep=',', index=True)
    
    file_path = os.path.join(
        folder_path,
        f"both_gender_preprocessed_data_data.csv")
    df.to_csv(file_path, sep=',', index=True)
  
else:
    healthy_save_data.save_preprocessed_data(df_female, population, mri_status, session, "female")
    healthy_save_data.save_preprocessed_data(df_male, population, mri_status, session, "male")
    healthy_save_data.save_preprocessed_data(df, population, mri_status, session, "both_gender")

print("===== Done! =====")
embed(globals(), locals())