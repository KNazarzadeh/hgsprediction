import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results.healthy import load_trained_model_results
from hgsprediction.load_data.healthy import load_healthy_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.save_results.healthy import save_hgs_predicted_results

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
data_set = sys.argv[10]
gender = sys.argv[11]
###############################################################################
best_model_trained = load_trained_model_results.load_best_model_trained(
    "healthy",
    "nonmri",
    int(confound_status),
    gender,
    feature_type,
    target,
    "linear_svm",
    n_repeats,
    n_folds,
    session,
    "training_set",
)

print(best_model_trained)
print("gender is :", gender)

##############################################################################
# Define main features and extra features:
features, extend_features = define_features(feature_type)
##############################################################################
# Define X as main features and y as target:
X = features
y = target
##############################################################################
# load data # Extract data based on main features, extra features, target for each session and mri status:
data_extracted = load_healthy_data.load_extracted_data_by_feature_and_target(
    population,
    mri_status,
    feature_type,
    target,
    session,
    gender,
    data_set,
)

##############################################################################
# Predict Handgrip strength (HGS) on X and y in dataframe
# With best trained model on non-MRI healthy controls data
df = predict_hgs(data_extracted.copy(), X, y, best_model_trained, target)

##############################################################################
# Print the final dataframe after adding predicted and delta HGS columns
print(df)

save_hgs_predicted_results(
    df,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)
print("===== Done! =====")
embed(globals(), locals())
