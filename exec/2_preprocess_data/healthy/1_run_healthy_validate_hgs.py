
import sys
import pandas as pd
from hgsprediction.load_data import healthy_load_data
from hgsprediction.data_preprocessing import HealthyDataPreprocessor
from hgsprediction.save_data import healthy_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

if mri_status == "nonmri":
    df = healthy_load_data.load_original_binned_train_data(population, mri_status)
elif mri_status == "mri":
    df = healthy_load_data.load_original_data(population, mri_status)

data_processor = HealthyDataPreprocessor(df, mri_status)
# CHECK HGS AVAILABILITY
df = data_processor.check_hgs_availability(df)
# DATA VALIDATION
df = data_processor.validate_handgrips(df)
# Remove all columns with all NaN values
df = data_processor.remove_nan_columns(df)

df_female = df[df["31-0.0"]==0.0]
df_male = df[df["31-0.0"]==1.0]

healthy_save_data.save_validate_hgs_data(df_female, population, mri_status,"female")
healthy_save_data.save_validate_hgs_data(df_male, population, mri_status,"male")
healthy_save_data.save_validate_hgs_data(df, population, mri_status,"both_gender")


print("===== Done! =====")
embed(globals(), locals())