
import sys
import pandas as pd
from hgsprediction.load_data import parkinson_load_data
from hgsprediction.save_data import parkinson_save_data
from hgsprediction.compute_target import parkinson_compute_target
from ptpython.repl import embed


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
target = sys.argv[3]
parkinson_type = sys.argv[4]
        
df = parkinson_load_data.load_preprocessed_data(population, mri_status, parkinson_type, "both_gender")

df = parkinson_compute_target.compute_target(df, mri_status, parkinson_type, target)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]

parkinson_save_data.save_preprocessed_data(df_female, population, mri_status, parkinson_type, "female")
parkinson_save_data.save_preprocessed_data(df_male, population, mri_status, parkinson_type, "male")
parkinson_save_data.save_preprocessed_data(df, population, mri_status, parkinson_type, "both_gender")

print("===== Done! =====")
embed(globals(), locals())