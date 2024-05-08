import math
import sys
import os
import numpy as np
import pandas as pd


from ptpython.repl import embed



all_file_path = "/data/project/stroke_ukb/knazarzadeh/GIT_repositories/motor_ukb/data_ukb/data_hgs/healthy/original_data/all_healthy/all_healthy.csv"

mri_file_path = "/data/project/stroke_ukb/knazarzadeh/GIT_repositories/motor_ukb/data_ukb/data_hgs/healthy/original_data/mri_healthy/mri_healthy.csv"

nonmri_file_path = "/data/project/stroke_ukb/knazarzadeh/GIT_repositories/motor_ukb/data_ukb/data_hgs/healthy/original_data/nonmri_healthy/nonmri_healthy.csv"

df_all = pd.read_csv(all_file_path, sep=',', low_memory=False)

df_mri = pd.read_csv(mri_file_path, sep=',', low_memory=False)

df_nonmri = pd.read_csv(nonmri_file_path, sep=',', low_memory=False)

df_all = df_all.rename(columns={"eid":"SubjectID"})
df_all = df_all.set_index("SubjectID")

df_mri = df_mri.rename(columns={"eid":"SubjectID"})
df_mri = df_mri.set_index("SubjectID")

df_nonmri = df_nonmri.rename(columns={"eid":"SubjectID"})
df_nonmri = df_nonmri.set_index("SubjectID")

print("===== Done! =====")
embed(globals(), locals())