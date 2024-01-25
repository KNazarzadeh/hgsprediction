import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.load_data import healthy_load_data
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results.healthy import save_spearman_correlation_results, \
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
###############################################################################
female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "female",
                                feature_type,
                                target,
                                f"{model_name}",
                                10,
                                5,
                            )

print(female_best_model_trained)

male_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "male",
                                feature_type,
                                target,
                                f"{model_name}",
                                10,
                                5,
                            )
print(male_best_model_trained)
##############################################################################
# load data
folder_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/preprocessed_data/nonmri_healthy/test_holdout_set/preprocessed_data/0_session_ukb"

file_path = os.path.join(folder_path, "both_gender_preprocessed_data.csv")

df = pd.read_csv(file_path, sep=',', index_col=0)

features = define_features(feature_type)
df_extracted = healthy_extract_data.extract_data(df,mri_status, features, target, 0)

rename_mapping = {col: col.replace("-0.0", '') for col in df.columns if "-0.0" in col}
df_extracted.rename(columns=rename_mapping, inplace=True)

X = features
y = target

df_female = df_extracted[df_extracted["gender"] == 0]
df_male = df_extracted[df_extracted["gender"] == 1]


df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)

print(df_female)
print(df_male)

df_both_gender = pd.concat([df_female, df_male], axis=0)
print(df_both_gender)


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