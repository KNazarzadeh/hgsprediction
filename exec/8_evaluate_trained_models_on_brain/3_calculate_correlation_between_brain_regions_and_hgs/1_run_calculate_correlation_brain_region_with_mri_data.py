import os
import sys
import pandas as pd
import numpy as np

from hgsprediction.load_data.brain_correlate.load_overlap_brain_data_with_mri_data import load_overlap_brain_data_with_mri_data
from hgsprediction.load_data.brain_correlate.load_removed_tiv_from_brain_data import load_removed_tiv_from_brain_data
from hgsprediction.brain_correlate.calculate_brain_correlation_with_hgs import calculate_brain_correlation_with_hgs
from hgsprediction.save_results.brain_correlate.save_brain_correlation_results import save_hgs_correlation_with_brain_regions_results
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
stats_correlation_type = sys.argv[14]
corr_target = sys.argv[15]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df_session_2 = load_overlap_brain_data_with_mri_data(
    brain_data_type,
    schaefer,
    "2",
    gender,)

df_session_3 = load_overlap_brain_data_with_mri_data(
    brain_data_type,
    schaefer,
    "3",
    gender,)
print(df_session_3)
###############################################################################
if len(df_session_3) > 0:
    df = pd.concat([df_session_2, df_session_3], axis=0)
else:
    df = df_session_2.copy()

##############################################################################
x_axis = df.iloc[:,:1088].columns
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
if corr_target == "hgs_true":
    y_axis = f"{target}"
elif corr_target == "hgs_predicted":
    y_axis = f"{target}_predicted"
elif corr_target == "hgs_corrected_predicted":
    y_axis = f"{target}_corrected_predicted"
elif corr_target == "hgs_delta":
    y_axis = f"{target}_delta(true-predicted)"
elif corr_target == "hgs_corrected_delta":
    y_axis = f"{target}_corrected_delta(true-predicted)"

##############################################################################
# Correlation with True HGS
df_corr, df_corr_significant, n_regions_survived = calculate_brain_correlation_with_hgs(df, y_axis, x_axis, stats_correlation_type)
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
session = '2_and_3'
save_hgs_correlation_with_brain_regions_results(
    df_corr,
    brain_data_type,
    schaefer,
    session,
    gender,
    corr_target,
)


print("n_regions_survived=", n_regions_survived)
print(x_axis)
print("===== Done! =====")
embed(globals(), locals())