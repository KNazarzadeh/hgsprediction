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
###############################################################################
female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                "female",
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
                            )

print(female_best_model_trained)

male_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                "male",
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
                            )
print(male_best_model_trained)
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
# load data
df = healthy_load_data.load_preprocessed_nonmri_test_data(population, mri_status, session, "both_gender")

features, extend_features = define_features(feature_type)

data_extracted = healthy_extract_data.extract_data(df, features, extend_features, target, mri_status, session)

X = features
y = target


df_female = data_extracted[data_extracted["gender"] == 0]
df_male = data_extracted[data_extracted["gender"] == 1]

# print("===== Done! =====")
# embed(globals(), locals())
df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)

print(df_female)
print(df_male)

df_both_gender = pd.concat([df_female, df_male], axis=0)
print(df_both_gender)
# print("===== Done! =====")
# embed(globals(), locals())

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
        "hgs_predicted_results",
    )
    
if(not os.path.isdir(folder_path)):
    os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"both_gender_hgs_predicted_results.csv")

df_both_gender.to_csv(file_path, sep=',', index=True)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"female_hgs_predicted_results.csv")

df_female.to_csv(file_path, sep=',', index=True)


# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"male_hgs_predicted_results.csv")

df_male.to_csv(file_path, sep=',', index=True)


print("===== Done! =====")
embed(globals(), locals())