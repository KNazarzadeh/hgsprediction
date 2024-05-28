
import pandas as pd
import numpy as np
import sys
####### Parse Input #######
from hgsprediction.input_arguments import parse_args, input_arguments
####### Load Train set #######
# Load Processed Train set (after data validation, feature engineering)
from hgsprediction.load_data import healthy_load_data
####### Data Extraction #######
from hgsprediction.extract_data import healthy_extract_data
####### Features Extraction #######
from hgsprediction.define_features import define_features

# IMPORT SAVE FUNCTIONS
from hgsprediction.save_data import save_extracted_data_by_feature_and_target

#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
session = sys.argv[5]
data_set = sys.argv[6]
gender = sys.argv[7]
###############################################################################

df = healthy_load_data.load_preprocessed_data(population, mri_status, session, gender, data_set)

features, extend_features = define_features(feature_type)

data_extracted = healthy_extract_data.extract_data(df, features, extend_features, feature_type, target, mri_status, session)

X = features
y = target
print(data_extracted)

save_extracted_data_by_feature_and_target(
    data_extracted,
    population,
    mri_status,
    feature_type,
    target,
    session,
    gender,
    data_set,
)

print("===== Done! =====")
embed(globals(), locals())

# Test removed missed features and target
df[[f"bmi-{session}.0", f"height-{session}.0", f"waist_to_hip_ratio-{session}.0", f"hgs_L+R-{session}.0"]].isna().sum()
idx = df[df[f"bmi-{session}.0"].isna()].index
df[df[f"height-{session}.0"].index.isin(idx)][f"height-{session}.0"].isna().sum()
df[df[f"waist_to_hip_ratio-{session}.0"].index.isin(idx)][f"waist_to_hip_ratio-{session}.0"].isna().sum()