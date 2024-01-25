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
import pickle

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
# female_best_model_trained = load_trained_models.load_best_model_trained(
#                                 "healthy",
#                                 "nonmri",
#                                 0,
#                                 "female",
#                                 feature_type,
#                                 target,
#                                 "linear_svm",
#                                 10,
#                                 5,
#                             )

# print(female_best_model_trained)

# male_best_model_trained = load_trained_models.load_best_model_trained(
#                                 "healthy",
#                                 "nonmri",
#                                 0,
#                                 "male",
#                                 feature_type,
#                                 target,
#                                 "linear_svm",
#                                 10,
#                                 5,
#                             )
# print(male_best_model_trained)
# ##############################################################################
# df_train = healthy_load_data.load_preprocessed_data(population, mri_status, session, "both_gender")

# features = define_features(feature_type)

# data_extracted = healthy_extract_data.extract_data(df_train, mri_status, features, target, session)


# df_female_data = data_extracted[data_extracted['gender']==0]
# df_male_data = data_extracted[data_extracted['gender']==1]

##############################################################################
# Assuming that you have already trained and instantiated the model as `model`

if model_name == "both":
    for model_name in ["linear_svm", "random_forest"]:
        for gender in ['female', 'male']:
            folder_path = os.path.join(
                "/data",
                "project",
                "stroke_ukb",
                "knazarzadeh",
                "project_hgsprediction",
                "results_hgsprediction",          
                f"{population}",
                f"{mri_status}",
                f"{feature_type}",
                f"{target}",
                "without_confound_removal",
                f"{model_name}",
                "10_repeats_5_folds",
                f"{gender}",
                "scores_trained",)

            # Define the csv file path to save
            file_path = os.path.join(
                folder_path,
                f"scores_trained.pkl")
            
            df = pd.read_pickle(file_path)
            if gender == 'female':
                df_female = df
                df_female.loc[:, "gender"]= "female"

            else:
                df_male = df
                df_male.loc[:, "gender"]= "male"

        df = pd.concat([df_female[['gender', 'test_score']], df_male[['gender', 'test_score']]])
        if model_name == "linear_svm":
            df_svm = df
        else:
            df_rf = df


print("===== Done! =====")
embed(globals(), locals())
##############################################################################
df_svm.loc[:, "model"]= "Linear SVM"
df_rf.loc[:, "model"]= "RF"

df_combined_models_scores = pd.concat([df_svm, df_rf])

custom_palette = {'male': '#069AF3', 'female': 'red'}

fig = plt.figure(figsize=(18,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 25,
                     "xtick.labelsize": 25})

ax = sns.set_style("whitegrid")
ax = sns.violinplot(data=df_combined_models_scores, x="model", y="test_score", hue='gender',
               palette=custom_palette, linewidth=3)

plt.title(f"Compare Linear SVM and Randomforest -> Anthropometrics+Age features,{target}", fontsize=20, fontweight="bold")

plt.xlabel("Gender", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")

# ymin, ymax = plt.ylim()
# y_step_value = 0.01
# plt.yticks(np.arange(round(ymin/0.01)*.01-y_step_value, round(ymax/0.01)*.01, 0.01), fontsize=18, weight='bold')

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig(f"both_gender_both_model_{target}_violin.png")


print("===== Done! =====")
embed(globals(), locals())