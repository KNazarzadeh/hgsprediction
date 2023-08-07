
import pandas as pd
import numpy as np
from hgsprediction.load_data.load_healthy import load_mri_data
from hgsprediction.data_preprocessing import DataPreprocessor

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())



population="healthy"

data = load_mri_data(population)

# Check the availa
hgs_left = "46"  # Handgrip_strength_(left)
hgs_right = "47"  # Handgrip_strength_(right)
df_tmp_mri = pd.DataFrame()
ses = 2
df = data.copy()
df_tmp_mri = df[
        ((~df[f'{hgs_left}-{ses}.0'].isna()) &
            (df[f'{hgs_left}-{ses}.0'] !=  0))
        & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
            (df[f'{hgs_right}-{ses}.0'] !=  0))
        ]

df_test = df_tmp_mri[df_tmp_mri.columns[~df_tmp_mri.columns.str.contains("-1.*|-3.*")]]

print("===== Done! =====")
embed(globals(), locals())
##############################################################################
data_processor = DataPreprocessor(df_test,mri_status="mri")

# Call all functions inside the class
# DATA VALIDATION
data = data_processor.validate_handgrips(data)
# FEATURE ENGINEERING
data = data_processor.calculate_qualification(data)
data = data_processor.calculate_waist_to_hip_ratio(data)
data = data_processor.sum_handgrips(data)
data = data_processor.calculate_left_hgs(data)
data = data_processor.calculate_right_hgs(data)
data = data_processor.sub_handgrips(data)
# data = data_processor.calculate_laterality_index(data)
# data = data_processor.calculate_neuroticism_score(data)
# data = data_processor.calculate_anxiety_score(data)
# data = data_processor.calculate_depression_score(data)
# data = data_processor.calculate_cidi_score(data)
# data = data_processor.preprocess_behaviours(data)
data = data_processor.remove_nan_columns(data)
