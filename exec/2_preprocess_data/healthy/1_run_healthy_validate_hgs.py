
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


df_train = healthy_load_data.load_original_binned_train_data(population, mri_status)

data_processor = HealthyDataPreprocessor(df_train, mri_status)
# CHECK HGS AVAILABILITY
df = data_processor.check_hgs_availability(df_train)
# DATA VALIDATION
df = data_processor.validate_handgrips(df)
print("===== Done! =====")
embed(globals(), locals())
healthy_save_data.save_validate_hgs_data(df, population, mri_status)
# Remove all columns with all NaN values
df = data_processor.remove_nan_columns(df)
healthy_save_data.save_preprocessed_train_data(df, population, mri_status)
