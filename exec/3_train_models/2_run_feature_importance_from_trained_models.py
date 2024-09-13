import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results.healthy import load_trained_model_results
####### Features Extraction #######
from hgsprediction.define_features import define_features

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
    population,
    mri_status,
    int(confound_status),
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    session,
    data_set,
)

print(best_model_trained)
print("gender is :", gender)

###############################################################################
features, extend_features = define_features(feature_type)

X = features
y = target
###############################################################################
# Get the feature names (assuming X is a list of feature names)
feature_names = X  # If X is a list of feature names
###############################################################################
# Access the model from the pipeline
if model_name == "linear_svm":
    model = best_model_trained.named_steps['linearsvrheuristicc_zscore']
    # Get the feature importances from the linearsvr model
    importances = model.coef_

elif model_name == "random_forest":
    model = best_model_trained.named_steps['rf']
    # Get the feature importances from the Random Forest model
    importances = model.feature_importances_

# Map feature names to feature importances
feature_importance = dict(zip(feature_names, importances))

# Display the feature importances
for feature, importance in feature_importance.items():
    print(f"Feature: {feature}, Importance: {importance}")

print("===== Done! =====")
embed(globals(), locals())