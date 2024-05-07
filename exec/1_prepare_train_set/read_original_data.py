import sys
import os
import pandas as pd
import numpy as np

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

folder_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/"

file_path = os.path.join(folder_path, "all_healthy/all_healthy.csv")

df_all = pd.read_csv(file_path, sep=',', low_memory=False)

###################################################################
file_path = os.path.join(folder_path, "nonmri_healthy/nonmri_healthy.csv")

df_nonmri = pd.read_csv(file_path, sep=',', low_memory=False)

###################################################################
file_path = os.path.join(folder_path, "mri_healthy/mri_healthy.csv")

df_mri = pd.read_csv(file_path, sep=',', low_memory=False)

###################################################################

print("All healthy subjects:")
print("howmany Females:", len(df_all[df_all['31-0.0']==0]))
print("howmany '%' Females:", round(len(df_all[df_all['31-0.0']==0])*100/len(df_all), 2))

print("howmany Males:", len(df_all[df_all['31-0.0']==1]))
print(df_all.describe())

print("nonMRI healthy subjects:")
print(df_nonmri.describe())

print("MRI healthy subjects:")
print(df_mri.describe())



print("===== Done! =====")
embed(globals(), locals())