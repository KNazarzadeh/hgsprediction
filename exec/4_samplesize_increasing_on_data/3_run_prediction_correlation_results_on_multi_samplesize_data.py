
import sys
import pandas as pd
import numpy as np

from hgsprediction.load_results.healthy.load_multi_samples_trained_models_results import load_scores_trained

#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
# Inputs : Required inputs
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set = sys.argv[9]
gender = sys.argv[10]
samplesize = sys.argv[11]
###############################################################################
session = "0"
###############################################################################
samplesize = samplesize + "_percent"
print(samplesize)
df = load_scores_trained(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    session,
    data_set,
    samplesize,
    )
###############################################################################
print("R2 Median =", "{:.3f}".format(np.median(df["test_r2"])))
print("R Score Median =", "{:.3f}".format(np.median(df["test_pearson_corr"])))
print("MAE Median =", "{:.3f}".format(np.median(df["test_neg_mean_absolute_error"]*-1)))

print("===== Done! End =====")
embed(globals(), locals())