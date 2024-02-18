import os
import sys
import pandas as pd
from hgsprediction.load_data import healthy_load_data
from hgsprediction.save_data import healthy_save_data
from hgsprediction.compute_target import healthy_compute_target
from ptpython.repl import embed


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
target = sys.argv[3]
session = sys.argv[4]
data_set = sys.argv[5]

# for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R", "hgs_dominant", "hgs_nondominant"]:
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
    
    file_path = os.path.join(
        folder_path,
        f"both_gender_preprocessed_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
    
else:
            
    df = healthy_load_data.load_preprocessed_data(population, mri_status, session, "both_gender")

df = healthy_compute_target.compute_target(df, mri_status, session, target)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]

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
        f"female_processed_data.csv")
    df_female.to_csv(file_path, sep=',', index=True)
    
    file_path = os.path.join(
        folder_path,
        f"male_processed_data.csv")
    df_male.to_csv(file_path, sep=',', index=True)
    
    file_path = os.path.join(
        folder_path,
        f"both_gender_processed_data.csv")
    df.to_csv(file_path, sep=',', index=True)
  
else:
    healthy_save_data.save_preprocessed_data(df_female, population, mri_status, session, "female")
    healthy_save_data.save_preprocessed_data(df_male, population, mri_status, session, "male")
    healthy_save_data.save_preprocessed_data(df, population, mri_status, session, "both_gender")

print("===== Done! =====")
embed(globals(), locals())