import pandas as pd
import sys

from hgsprediction.load_data import parkinson_load_data
from hgsprediction.data_preprocessing import parkinson_data_preprocessor
from hgsprediction.save_data import parkinson_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

df = parkinson_load_data.load_original_data(population=population, mri_status=mri_status)

df_premanifest = df[df["Manifest?"]==0]
df_manifest = df[df["Manifest?"]==1]

data_processor_premanifest = parkinson_data_preprocessor.ParkinsonDataPreprocessor(df_premanifest, mri_status, session="0")
data_processor_manifest = parkinson_data_preprocessor.ParkinsonDataPreprocessor(df_manifest, mri_status, session="2")

# CHECK HGS AVAILABILITY
df_premanifest = data_processor_premanifest.check_hgs_availability(df_premanifest)
# DATA VALIDATION
df_premanifest = data_processor_premanifest.validate_handgrips(df_premanifest)
# Remove all columns with all NaN values
df_premanifest = data_processor_premanifest.remove_nan_columns(df_premanifest)

# CHECK HGS AVAILABILITY
df_manifest = data_processor_manifest.check_hgs_availability(df_manifest)

# DATA VALIDATION
df_manifest = data_processor_manifest.validate_handgrips(df_manifest)
# Remove all columns with all NaN values
df_manifest = data_processor_manifest.remove_nan_columns(df_manifest)

df_female_premanifest = df_premanifest[df_premanifest["31-0.0"]==0.0]
df_male_premanifest = df_premanifest[df_premanifest["31-0.0"]==1.0]


df_female_manifest = df_manifest[df_manifest["31-0.0"]==0.0]
df_male_manifest = df_manifest[df_manifest["31-0.0"]==1.0]


parkinson_save_data.save_validate_hgs_data(df_female_premanifest, population, mri_status, "premanifest", "female")
parkinson_save_data.save_validate_hgs_data(df_male_premanifest, population, mri_status, "premanifest", "male")
parkinson_save_data.save_validate_hgs_data(df_premanifest, population, mri_status, "premanifest", "both_gender")


parkinson_save_data.save_validate_hgs_data(df_female_manifest, population, mri_status, "manifest", "female")
parkinson_save_data.save_validate_hgs_data(df_male_manifest, population, mri_status, "manifest", "male")
parkinson_save_data.save_validate_hgs_data(df_manifest, population, mri_status, "manifest", "both_gender")

print("===== Done! =====")
embed(globals(), locals())
