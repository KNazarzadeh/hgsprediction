import os
import pandas as pd
import numpy as np
import sys


from hgsprediction.load_results import load_trained_models
from hgsprediction.load_data import healthy_load_data
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_pearson_hgs_correlation
from hgsprediction.save_results.healthy import save_correlation_results, \
                                               save_hgs_predicted_results
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


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
gender = sys.argv[10]
###############################################################################
best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                gender,
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
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
# load data
df = healthy_load_data.load_preprocessed_nonmri_test_data(population, mri_status, session, gender)

##############################################################################
# Extract data based on main features, extra features, target for each session and mri status:
data_extracted = healthy_extract_data.extract_data(df, features, extend_features, feature_type, target, mri_status, session)
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
# Predict Handgrip strength (HGS) on X and y in dataframe
# With best trained model on non-MRI healthy controls data
df = predict_hgs(data_extracted, X, y, best_model_trained, target)

##############################################################################
# Print the final dataframe after adding predicted and delta HGS columns
print(df)

# Assuming that you have already trained and instantiated the model as `model`
folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",  
        "results_hgsprediction",
        f"{population}",
        "nonmri_test_holdout_set",
        f"{feature_type}",
        f"{target}",
        f"{model_name}",
        "hgs_corrected_predicted_results",
    )
    
if(not os.path.isdir(folder_path)):
    os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_predicted_results.csv")

df.to_csv(file_path, sep=',', index=True)

print("===== Done! =====")
embed(globals(), locals())