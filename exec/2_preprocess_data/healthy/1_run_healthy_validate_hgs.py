
import sys
import os
import pandas as pd
import numpy as np
from hgsprediction.load_data.healthy import load_healthy_data
from hgsprediction.data_preprocessing import HealthyDataPreprocessor
from hgsprediction.save_data.healthy import save_healthy_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
session = sys.argv[3]
data_set = sys.argv[4]

if mri_status == "nonmri":
    if data_set == "training_set":
        df_original = load_healthy_data.load_original_binned_train_data(population, mri_status)
    elif data_set == "holdout_test_set":
        df_original = load_healthy_data.load_original_nonmri_test_data(population, mri_status)
elif mri_status == "mri":
    df_original = load_healthy_data.load_original_data(population, mri_status)
###############################################################################
data_processor = HealthyDataPreprocessor(df_original, mri_status, session)
df = data_processor.define_handedness(df_original)

# CHECK HGS AVAILABILITY
df = data_processor.remove_missing_hgs(df)

# DATA VALIDATION
df = data_processor.validate_handgrips(df)

# Remove all columns with all NaN values
df = data_processor.remove_nan_columns(df)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]

save_healthy_data.save_validate_hgs_data(df_female, population, mri_status, session, "female", data_set)
save_healthy_data.save_validate_hgs_data(df_male, population, mri_status, session, "male", data_set)


print("===== Done! =====")
embed(globals(), locals())