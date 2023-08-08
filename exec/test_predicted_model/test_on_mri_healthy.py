
import pandas as pd
import numpy as np
import sys
import os
####### Load Train set #######
from hgsprediction.input_arguments import parse_args, input_arguments

from hgsprediction.load_trained_model import load_best_model_trained
from hgsprediction.load_data.load_healthy import load_mri_data_for_anthropometrics
from hgsprediction.data_extraction import data_extractor, run_data_extraction
from hgsprediction.extract_features import features_extractor
from hgsprediction.extract_target import target_extractor
from hgsprediction.LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model_name, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)

###############################################################################
df_test = load_mri_data_for_anthropometrics(population)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
data_extracted = run_data_extraction.data_extractor(df_test, mri_status, gender, feature_type, target)
print("===== Done! =====")
embed(globals(), locals())
X = features_extractor(data_extracted, mri_status, feature_type)
y = target_extractor(data_extracted, target)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
best_model_trained = load_best_model_trained(
                                population,
                                gender,
                                feature_type,
                                target,
                                confound_status,
                                model_name,
                                cv_repeats_number,
                                cv_folds_number,
                            )

###############################################################################
print("===== Done! =====")
embed(globals(), locals())
y_true = data_extracted[y]
y_pred = best_model_trained.predict(data_extracted[X])

# new_data["actual_hgs"] = y_true
# new_data["predicted_hgs"] = y_pred
# new_data["hgs_diff"] = y_true - y_pred
# mae = format(mean_absolute_error(y_true, y_pred), '.2f')
# corr = format(np.corrcoef(y_pred, y_true)[1, 0], '.2f')
# score = format(r2_score(y_true, y_pred), '.2f')