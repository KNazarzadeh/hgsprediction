import sys
import pandas as pd
from hgsprediction.load_data import healthy_load_data
from hgsprediction.save_data import healthy_save_data
from hgsprediction.compute_extra_column import healthy_compute_extra_column
from ptpython.repl import embed


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
extra_column = sys.argv[3]
        
df = healthy_load_data.load_preprocessed_data(population, mri_status, "both_gender")

df = healthy_compute_extra_column.compute_extra_column(df, mri_status, extra_column)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]
print("===== Done! =====")
embed(globals(), locals())
healthy_save_data.save_preprocessed_data(df_female, population, mri_status, "female")
healthy_save_data.save_preprocessed_data(df_male, population, mri_status, "male")
healthy_save_data.save_preprocessed_data(df, population, mri_status, "both_gender")

print("===== Done! =====")
embed(globals(), locals())