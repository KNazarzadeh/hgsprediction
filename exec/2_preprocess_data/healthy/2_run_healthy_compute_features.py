import sys
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
feature_type = sys.argv[3]
session = sys.argv[4]

df = healthy_load_data.load_validate_hgs_data(population, mri_status, session, "both_gender")

df = healthy_compute_features.compute_features(df, mri_status, feature_type, session)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]
print("===== Done! =====")
embed(globals(), locals())
healthy_save_data.save_preprocessed_data(df_female, population, mri_status, session, "female")
healthy_save_data.save_preprocessed_data(df_male, population, mri_status, session, "male")
healthy_save_data.save_preprocessed_data(df, population, mri_status, session, "both_gender")

print("===== Done! =====")
embed(globals(), locals())