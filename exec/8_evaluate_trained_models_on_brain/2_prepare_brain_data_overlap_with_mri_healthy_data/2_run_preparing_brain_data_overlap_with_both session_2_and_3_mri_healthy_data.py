import os
import sys
import pandas as pd
import numpy as np

from hgsprediction.load_data.brain_correlate.load_overlap_brain_data_with_mri_data import load_overlap_brain_data_with_mri_data
from hgsprediction.load_data.brain_correlate.load_removed_tiv_from_brain_data import load_removed_tiv_from_brain_data
from hgsprediction.save_data.brain_correlate.save_overlap_brain_data_with_mri_data import save_overlap_brain_data_with_mri_data

####### Features Extraction #######
from hgsprediction.define_features import define_features
#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set =sys.argv[9]
brain_data_type = sys.argv[10]
tiv_status = sys.argv[11]
schaefer = sys.argv[12]
gender = sys.argv[13]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
features, extend_features = define_features(feature_type)

X = features
y = target
###############################################################################
df_overlap_session_2 = load_overlap_brain_data_with_mri_data(
    brain_data_type,
    schaefer,
    "2",
    gender,)

df_overlap_session_3 = load_overlap_brain_data_with_mri_data(
    brain_data_type,
    schaefer,
    "3",
    gender,)

###############################################################################
df_overlap_session_3 = df_overlap_session_3[~df_overlap_session_3.index.isin(df_overlap_session_2.index)]

##############################################################################
if tiv_status == "without_tiv":
    df_brain_without_tiv = load_removed_tiv_from_brain_data(brain_data_type, schaefer)
##############################################################################
extract_cols = df_brain_without_tiv.columns.tolist() + X + [col for col in df_overlap_session_2.columns if target in col]
##############################################################################
df_merged = pd.concat([df_overlap_session_2[extract_cols], df_overlap_session_3[extract_cols]], axis=0)

##############################################################################
df_merged = df_merged[df_merged.index.isin(df_brain_without_tiv.index)]

##############################################################################

session = "2_and_3"
##############################################################################
print("===== Done! =====")
embed(globals(), locals())
save_overlap_brain_data_with_mri_data(
    df_merged,
    brain_data_type,
    schaefer,
    session,
    gender,
)

print("===== Done! =====")
embed(globals(), locals())