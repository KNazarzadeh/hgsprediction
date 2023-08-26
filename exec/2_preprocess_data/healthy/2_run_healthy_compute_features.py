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

df = healthy_load_data.load_preprocessed_train_data(population, mri_status)

df = healthy_compute_features.compute_features(df, mri_status, feature_type)
print("===== Done! =====")
embed(globals(), locals())
healthy_save_data.save_preprocessed_train_data(df, population, mri_status)
