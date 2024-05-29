import sys
import os
import pandas as pd
from hgsprediction.load_data import load_healthy_data
from hgsprediction.compute_features import healthy_compute_features
from hgsprediction.save_data import save_healthy_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
session = sys.argv[4]
data_set = sys.argv[5]
gender = sys.argv[6]


df = load_healthy_data.load_validate_hgs_data(population, mri_status, session, gender, data_set)

# df = healthy_compute_features.compute_features(df, mri_status, feature_type, session)
df = healthy_compute_features.compute_features(df, feature_type, mri_status, session)

save_healthy_data.save_preprocessed_data(df, population, mri_status, session, gender, data_set)

print("===== Done! =====")
embed(globals(), locals())