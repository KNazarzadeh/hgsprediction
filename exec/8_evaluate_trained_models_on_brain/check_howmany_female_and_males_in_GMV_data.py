import os
import sys
# import argparse
import pandas as pd

from hgsprediction.load_data.brain_correlate.load_removed_tiv_from_brain_data import load_removed_tiv_from_brain_data
from hgsprediction.load_data.brain_correlate import load_original_brain_data


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
session = sys.argv[3]
brain_data_type = sys.argv[4]
schaefer = sys.argv[5]
tiv_status = sys.argv[6]
############################################

folder_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/all_original_MRI_data_to_check_for_gender_GMV/"
file_path = os.path.join(folder_path, "all_original_MRI_data_to_check_for_gender_GMV.csv")

df = pd.read_csv(file_path, sep=',')

df = df.rename(columns={"eid":"SubjectID"})
df = df.set_index("SubjectID")
############################################


df_brain = load_original_brain_data(brain_data_type, schaefer)
# Remove the "seu-" prefix from the index values
df_brain.index = [int(i.replace('sub-', '')) for i in df_brain.index]


if tiv_status == "without_tiv":
    df_brain_without_tiv = load_removed_tiv_from_brain_data(session, brain_data_type, schaefer)
    
############################################


df_overlap = df[df.index.isin(df_brain_without_tiv.index)]
df_female = df_overlap[df_overlap['31-0.0']==0]
df_male = df_overlap[df_overlap['31-0.0']==1]

print("Female number=", len(df_female))
print("Male number=", len(df_male))

print("===== Done! =====")
embed(globals(), locals())
