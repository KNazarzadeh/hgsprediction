
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
        
df = healthy_load_data.load_preprocessed_data(population, mri_status, session, "both_gender")

df = healthy_compute_target.compute_target(df, mri_status, session, target)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]

healthy_save_data.save_preprocessed_data(df_female, population, mri_status, session, "female")
healthy_save_data.save_preprocessed_data(df_male, population, mri_status, session, "male")
healthy_save_data.save_preprocessed_data(df, population, mri_status, session, "both_gender")

print("===== Done! =====")
embed(globals(), locals())