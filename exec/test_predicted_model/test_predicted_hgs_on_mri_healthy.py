
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
from hgsprediction.save_results import save_extracted_mri_data, \
                                       save_tested_mri_data
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

###############################################################################
data_extracted = run_data_extraction.data_extractor(df_test, mri_status, gender, feature_type, target)
print(data_extracted)

X = features_extractor(data_extracted, mri_status, feature_type)
y = target_extractor(data_extracted, target)
print(X)
print(y)
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
print(best_model_trained)
###############################################################################
y_true = data_extracted[y]
y_pred = best_model_trained.predict(data_extracted[X])

data_tested = data_extracted.copy()
data_tested["hgs_predicted"] = y_pred
data_tested["hgs_actual-predicted"] = y_true - y_pred

print(data_tested)
###############################################################################
save_extracted_mri_data(
    data_extracted,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
)
###############################################################################
save_tested_mri_data(
    data_tested,
    population,
    mri_status,
    gender,
    feature_type,
    target,
    confound_status,
    model_name,
    cv_repeats_number,
    cv_folds_number,
)

print("===== Done! =====")
