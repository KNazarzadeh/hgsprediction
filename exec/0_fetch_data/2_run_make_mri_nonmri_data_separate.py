
import pandas as pd
import numpy as np
import sys
import os

from hgsprediction.load_data import disorder_load_data


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

if mri_status in ["all", "mri"]:
    df = disorder_load_data.load_original_data(population, mri_status)
    df.index.names = ["SubjectID"]
    
elif mri_status == "nonmri":
    df_all_original = disorder_load_data.load_original_data(population, "all")
    df_all_original.index.names = ["SubjectID"]
    df_mri_original = disorder_load_data.load_original_data(population, "mri")
    df_mri_original.index.names = ["SubjectID"]

    df = df_all_original[~df_all_original.index.isin(df_mri_original.index)]

###############################################################################
#Create folders and file names to save the fetched data on Juseless
# ----------- On Juseless ----------#
# ----- The main folder is the directory folder to save all sub-folders
# ----------------------------------#
main_folder_path = os.path.join(
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

# ----- make the directory folder if it has not created before
if(not os.path.isdir(main_folder_path)):
    os.mkdir(main_folder_path)

# ----- Define the output .csv file name and join to the directory
file_path = os.path.join(
        main_folder_path,
        f"{mri_status}_{population}.csv"
        )
    
df.to_csv(file_path, sep=',', index=True)

print("===== Done! =====")
embed(globals(), locals())