
import sys
import numpy as np
import pandas as pd
import os
from psmpy import PsmPy
from hgsprediction.load_results import healthy
from hgsprediction.load_data import healthy_load_data
from hgsprediction.load_results import stroke
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import predict_hgs
from scipy.stats import spearmanr



from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

###############################################################################
female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "female",
                                feature_type,
                                target,
                                "linear_svm",
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
                                "linear_svm",
                                10,
                                5,
                            )
print(male_best_model_trained)
##############################################################################

if target == "hgs_L+R":
    target_label = "HGS (Left+Right)"
elif target == "hgs_left":
    target_label = "HGS (Left)"
elif target == "hgs_right":
    target_label = "HGS (Right)"

df_mri_1st_scan = healthy.load_hgs_predicted_results("healthy",
    "mri",
    "linear_svm",
    "anthropometrics_age",
    f"{target}",
    "both_gender",
    session="2",
)
df_healthy = df_mri_1st_scan.loc[:, ["gender", "1st_scan_age", "1st_scan_bmi",  "1st_scan_height",  "1st_scan_waist_to_hip_ratio", f"1st_scan_{target}"]]

df_healthy.rename(columns={"1st_scan_age":"age", "1st_scan_bmi":"bmi",  "1st_scan_height":"height",  "1st_scan_waist_to_hip_ratio":"waist_to_hip_ratio",
                           "1st_scan_handedness":"handedness", f"1st_scan_{target}":f"{target}"}, inplace=True)

df_healthy.loc[:, "disease"] = 0

###############################################################################

stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_stroke = stroke.load_hgs_predicted_results("stroke", mri_status, session_column, model_name, feature_type, target, "both_gender")
df_stroke.loc[:, "disease"] = 1

df_pre_stroke = df_stroke.loc[:, ["gender", "1st_pre-stroke_age", "1st_pre-stroke_bmi",  "1st_pre-stroke_height",  "1st_pre-stroke_waist_to_hip_ratio", f"1st_pre-stroke_{target}", "disease"]]
df_pre_stroke.rename(columns={"1st_pre-stroke_age":"age", "1st_pre-stroke_bmi":"bmi",  "1st_pre-stroke_height":"height",  "1st_pre-stroke_waist_to_hip_ratio":"waist_to_hip_ratio", 
                              "1st_pre-stroke_handedness":"handedness", f"1st_pre-stroke_{target}":f"{target}"}, inplace=True)

df_post_stroke = df_stroke.loc[:, ["gender", "1st_post-stroke_age", "1st_post-stroke_bmi",  "1st_post-stroke_height",  "1st_post-stroke_waist_to_hip_ratio", f"1st_post-stroke_{target}", "disease"]]
df_post_stroke.rename(columns={"1st_post-stroke_age":"age", "1st_post-stroke_bmi":"bmi",  "1st_post-stroke_height":"height",  "1st_post-stroke_waist_to_hip_ratio":"waist_to_hip_ratio",
                               "1st_post-stroke_handedness":"handedness", f"1st_post-stroke_{target}":f"{target}"}, inplace=True)

###############################################################################
df_pre = pd.concat([df_healthy, df_pre_stroke], axis=0)
df_pre.insert(0, "index", df_pre.index)

df_post = pd.concat([df_healthy, df_post_stroke], axis=0)
df_post.insert(0, "index", df_post.index)

data = df_pre[df_pre['gender']==0]
##############################################################################
# Define the covariates you want to use for matching
# covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]
covariates = ["bmi", "height", "waist_to_hip_ratio", "age"]

##############################################################################

# Fit a logistic regression model to estimate propensity scores
propensity_model = LogisticRegression()
propensity_model.fit(data[covariates], data['disease'])
propensity_scores = propensity_model.predict_proba(data[covariates])[:, 1]
data['propensity_scores'] = propensity_scores

# Create a DataFrame for diseaseed and control groups
disease_group = data[data['disease'] == 1]
control_group = data[data['disease'] == 0]

matched_pairs = pd.DataFrame({'disease_index': disease_group.index})
matched_data = list()
matched_patients = pd.DataFrame()
matched_controls = pd.DataFrame()
unmatched_controls = pd.DataFrame()
unmatched_patients = pd.DataFrame()
# Define the range of k from 1 to n
n = 10  # You can change this to the desired value of n
for k in range(1, n + 1):
    # Fit a Nearest Neighbors model on the control group with the current k
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(control_group[covariates])

    # Find the k nearest neighbors for each diseased unit
    distances, indices = knn.kneighbors(disease_group[covariates])
    
    matched_pairs_tmp = pd.DataFrame({
        f'control_index_{k}': control_group.index[indices[:,k-1].flatten()],
        f'distance_{k}': distances[:,k-1].flatten(),
        f'propensity_score_{k}': propensity_scores[indices[:,k-1].flatten()]
    })
    
    matched_pairs = pd.concat([matched_pairs, matched_pairs_tmp], axis=1)
    
    matched_data_tmp = disease_group.reset_index(drop=True).join(control_group.iloc[indices[:,k-1].flatten()].reset_index(drop=True), lsuffix="_disease", rsuffix="_control")
    matched_data_tmp['distance'] = matched_pairs[f'distance_{k}'].values

    matched_data.append(matched_data_tmp)
    
    print(matched_data)
    
    matched_patients[f'propensity_scores_disease_{k}'] = matched_data[k-1]['propensity_scores_disease']
    matched_controls[f'propensity_scores_disease_{k}'] = matched_data[k-1]['propensity_scores_control']
    unmatched_controls[f'propensity_scores_disease_{k}'] = control_group[~control_group.index.isin(matched_data[k-1]['index_control'])].loc[:, 'propensity_scores']
    unmatched_patients[f'propensity_scores_disease_{k}'] = disease_group[~disease_group.index.isin(matched_data[k-1]['index_disease'])].loc[:, 'propensity_scores']
    
print(matched_pairs)

print("===== Done! =====")
embed(globals(), locals())



# print("===== Done! =====")
# embed(globals(), locals())
# Use the matched pairs to create the matched data
control_samples = control_group.iloc[indices.flatten()]            
        
##############################################################################
##############################################################################

print("===== Done! =====")
embed(globals(), locals())