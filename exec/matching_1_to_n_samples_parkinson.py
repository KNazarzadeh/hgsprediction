import math
import sys
import numpy as np
import pandas as pd
import os
from psmpy import PsmPy
from hgsprediction.load_results import healthy
from hgsprediction.load_data import healthy_load_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import predict_hgs
from scipy.stats import spearmanr
from scipy.stats import zscore
from hgsprediction.load_results import parkinson


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
df_mri = df_mri_1st_scan.loc[:, ["gender", "1st_scan_age", "1st_scan_bmi",  "1st_scan_height",  "1st_scan_waist_to_hip_ratio", f"1st_scan_{target}"]]

df_mri.rename(columns={"1st_scan_age":"age", "1st_scan_bmi":"bmi",  "1st_scan_height":"height",  "1st_scan_waist_to_hip_ratio":"waist_to_hip_ratio",
                           "1st_scan_handedness":"handedness", f"1st_scan_{target}":f"{target}"}, inplace=True)

df_female_mri = df_mri[df_mri['gender']==0]
df_male_mri = df_mri[df_mri['gender']==1]
threshold = 3

z_scores_female = zscore(df_female_mri[["age", "bmi", "height", "waist_to_hip_ratio", f"{target}"]])
z_score_df_female = pd.DataFrame(z_scores_female, columns=["age", "bmi", "height", "waist_to_hip_ratio", f"{target}"])
outliers_female = (z_score_df_female > threshold) | (z_score_df_female < -threshold)
# Remove outliers
df_no_outliers_female = z_score_df_female[~outliers_female.any(axis=1)]
df_outliers_female = z_score_df_female[outliers_female.any(axis=1)]

z_scores_male = zscore(df_male_mri[["age", "bmi", "height", "waist_to_hip_ratio", f"{target}"]])
z_score_df_male = pd.DataFrame(z_scores_male, columns=["age", "bmi", "height", "waist_to_hip_ratio", f"{target}"])
outliers_male = (z_score_df_male > threshold) | (z_score_df_male < -threshold)
# Remove outliers
df_no_outliers_male = z_score_df_male[~outliers_male.any(axis=1)]
df_outliers_male = z_score_df_male[outliers_male.any(axis=1)]

df_healthy_female = df_female_mri[df_female_mri.index.isin(df_no_outliers_female.index)]
df_healthy_male = df_male_mri[df_male_mri.index.isin(df_no_outliers_male.index)]

df_healthy = pd.concat([df_healthy_female, df_healthy_male])
df_healthy.loc[:, "disease"] = 0
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

df_parkinson_premanifest = parkinson.load_hgs_predicted_results("parkinson", mri_status, model_name, feature_type, target, "both_gender", "premanifest")
df_parkinson_manifest = parkinson.load_hgs_predicted_results("parkinson", mri_status, model_name, feature_type, target, "both_gender", "manifest")
    
df_parkinson_premanifest.loc[:, "disease"] = 1
df_parkinson_manifest.loc[:, "disease"] = 1

df_parkinson_premanifest = df_parkinson_premanifest.loc[:, ["gender", "age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}", "disease"]]

df_parkinson_manifest = df_parkinson_manifest.loc[:, ["gender", "age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}", "disease"]]
###############################################################################
df_premanifest = pd.concat([df_healthy, df_parkinson_premanifest], axis=0)
df_premanifest.insert(0, "index", df_premanifest.index)

df_manifest = pd.concat([df_healthy, df_parkinson_manifest], axis=0)
df_manifest.insert(0, "index", df_manifest.index)

##############################################################################
# Define the covariates you want to use for matching
# covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]
covariates = ["bmi", "height", "waist_to_hip_ratio", "age"]
X = covariates
y = target
custom_palette = sns.color_palette(['#a851ab', '#005c95'])  # You can use any hex color codes you prefer
fig, ax = plt.subplots(2,2, figsize=(12,12))
for parkinson_type in ["premanifest", "manifest"]:
    if parkinson_type == "premanifest":
        df = df_premanifest.copy()
        df_parkinson = df_premanifest[df_premanifest['disease']==1]
        axj=0
    elif parkinson_type == "manifest":
        df = df_manifest.copy()
        df_parkinson = df_manifest[df_manifest['disease']==1]
        axj=1
##############################################################################
    control_samples_female = pd.DataFrame()
    control_samples_male = pd.DataFrame()
    for gender in ["Female", "Male"]:
        if gender == "Female":
            data = df[df["gender"]==0]
            df_female_parkinson = df_parkinson[df_parkinson["gender"]==0]
            print(gender)
        elif gender == "Male":    
            data = df[df["gender"]==1]
            df_male_parkinson = df_parkinson[df_parkinson["gender"]==1]
        # Fit a logistic regression model to estimate propensity scores
        propensity_model = LogisticRegression()
        propensity_model.fit(data[covariates], data['disease'])
        propensity_scores = propensity_model.predict_proba(data[covariates])[:, 1]
        data['propensity_scores'] = propensity_scores

        # Create a DataFrame for diseaseed and control groups
        disease_group = data[data['disease'] == 1]
        control_group = data[data['disease'] == 0]

        matched_pairs = pd.DataFrame({'disease_index': disease_group.index})
        matched_data = pd.DataFrame()
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
            # matched_data = matched_data.append(matched_data_tmp)
            matched_data = pd.concat([matched_data, matched_data_tmp], axis=0)
            if gender == "Female":
                control_samples_female = pd.concat([control_samples_female, control_group.iloc[indices[:,k-1].flatten()]], axis=0)
                df_female = control_samples_female.copy()
                df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
                df_female_parkinson = predict_hgs(df_female_parkinson, X, y, female_best_model_trained, target)
                corr_female_control = spearmanr(df_female[f"{target}_predicted"], df_female[f"{target}_actual"])[0]
                corr_female_parkinson = spearmanr(df_female_parkinson[f"{target}_predicted"], df_female_parkinson[f"{target}_actual"])[0]
            elif gender == "Male":                
                control_samples_male = pd.concat([control_samples_male, control_group.iloc[indices[:,k-1].flatten()]], axis=0)
                df_male = control_samples_male.copy()
                df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)
                df_male_parkinson = predict_hgs(df_male_parkinson, X, y, male_best_model_trained, target)
                corr_male_control = spearmanr(df_male[f"{target}_predicted"], df_male[f"{target}_actual"])[0]
                corr_male_parkinson = spearmanr(df_male_parkinson[f"{target}_predicted"], df_male_parkinson[f"{target}_actual"])[0]
        print(matched_data)
        print(matched_pairs)

    df_both_gender = pd.concat([df_female, df_male], axis=0)
    df_both_parkinson = pd.concat([df_female_parkinson, df_male_parkinson], axis=0)
    corr_control = spearmanr(df_both_gender[f"{target}_predicted"], df_both_gender[f"{target}_actual"])[0]
    corr_parkinson = spearmanr(df_both_parkinson[f"{target}_predicted"], df_both_parkinson[f"{target}_actual"])[0]
    text_control = 'r= ' + str(format(corr_control, '.3f'))
    text_parkinson = 'r= ' + str(format(corr_parkinson, '.3f'))
    text_control_female = 'r= ' + str(format(corr_female_control, '.3f'))
    text_parkinson_female = 'r= ' + str(format(corr_female_parkinson, '.3f'))
    text_control_male = 'r= ' + str(format(corr_male_control, '.3f'))
    text_parkinson_male = 'r= ' + str(format(corr_male_parkinson, '.3f'))
    print(df_both_gender)
    print(df_both_parkinson)
##############################################################################
##############################################################################
    print(axj)
    sns.regplot(data=df_female, x=f"{target}_actual", y=f"{target}_predicted", ax=ax[0][axj], scatter_kws={"color": "#a851ab"}, line_kws={"color": "darkgrey"})
    sns.regplot(data=df_male, x=f"{target}_actual", y=f"{target}_predicted", ax=ax[0][axj], scatter_kws={"color": "#005c95"}, line_kws={"color": "darkgrey"})
    # sns.scatterplot(data=df_both_gender, x=f"{target}_actual", y=f"{target}_predicted", hue="gender", ax=ax[0][axj], palette = custom_palette, legend=False)
    ax[0][axj].set_xlim(math.floor(df_both_gender[f"{target}_actual"].min()/10)*10, math.ceil(df_both_gender[f"{target}_actual"].max()/10)*10+10)
    ax[0][axj].set_ylim(math.floor(df_both_gender[f"{target}_predicted"].min()/10)*10, math.ceil(df_both_gender[f"{target}_predicted"].max()/10)*10+10)
    
    sns.regplot(data=df_female_parkinson, x=f"{target}_actual", y=f"{target}_predicted", ax=ax[1][axj], scatter_kws={"color": "#a851ab"}, line_kws={"color": "darkgrey"})
    sns.regplot(data=df_male_parkinson, x=f"{target}_actual", y=f"{target}_predicted", ax=ax[1][axj], scatter_kws={"color": "#005c95"}, line_kws={"color": "darkgrey"})
    
    # sns.scatterplot(data=df_both_parkinson, x=f"{target}_actual", y=f"{target}_predicted", hue="gender", ax=ax[1][axj], palette = custom_palette, legend=False)
    ax[1][axj].set_xlim(math.floor(df_both_parkinson[f"{target}_actual"].min()/10)*10, math.ceil(df_both_parkinson[f"{target}_actual"].max()/10)*10+10)
    ax[1][axj].set_ylim(math.floor(df_both_parkinson[f"{target}_predicted"].min()/10)*10, math.ceil(df_both_parkinson[f"{target}_predicted"].max()/10)*10+10)
    
    legend1 = ax[0][axj].legend()
    legend2 = ax[1][axj].legend()
    # Remove the legend from the axis
    legend1.remove()
    legend2.remove()
    for i in range(0,2):
        ax[i][axj].set_ylabel("Predicted HGS", fontsize=12, fontweight="bold")
        if i == 0:
            ax[i][axj].set_xlabel("")
        else:
            ax[i][axj].set_xlabel("True HGS", fontsize=12, fontweight="bold")

    if axj == 0:
        ax[0][axj].set_title(f"Control Samples - {text_control}\n(female_corr:{text_control_female}, male_corr:{text_control_male})", fontsize=10, fontweight="bold")
        ax[1][axj].set_title(f"Premanifest parkinson - {text_parkinson}\n(female_corr:{text_parkinson_female}, male_corr:{text_parkinson_male})", fontsize=10, fontweight="bold")
    elif axj == 1:
        ax[0][axj].set_title(f"Control Samples - {text_control}\n(female_corr:{text_control_female}, male_corr:{text_control_male})", fontsize=10, fontweight="bold")
        ax[1][axj].set_title(f"Manifest parkinson - {text_parkinson}\n(female_corr:{text_parkinson_female}, male_corr:{text_parkinson_male})", fontsize=10, fontweight="bold")
            
plt.suptitle(f"Control smaples vs patients - {target_label}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
plt.savefig(f"Scatterplot_control_1_10_all_samples_predicted_{target}_PD_mri.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())