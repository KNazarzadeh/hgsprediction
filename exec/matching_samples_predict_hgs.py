
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

##############################################################################
# Define the covariates you want to use for matching
# covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]
covariates = ["bmi", "height", "waist_to_hip_ratio", "age"]
custom_palette = sns.color_palette(['#a851ab', '#005c95'])  # You can use any hex color codes you prefer
fig, ax = plt.subplots(2,2, figsize=(12,12))
for stroke_cohort in ["pre-stroke", "post-stroke"]:
    if stroke_cohort == "pre-stroke":
        df = df_pre.copy()
        df_stroke = df_pre[df_pre['disease']==1]
        axj = 0
        print(stroke_cohort)
    elif stroke_cohort == "post-stroke":
        df = df_post.copy()
        df_stroke = df_post[df_post['disease']==1]
        axj = 1        
##############################################################################
    for gender in ["Female", "Male"]:
        if gender == "Female":
            data = df[df["gender"]==0]
            df_female_stroke = df_stroke[df_stroke["gender"]==0]
            print(gender)
            # print("===== Done! =====")
            # embed(globals(), locals())
        elif gender == "Male":    
            data = df[df["gender"]==1]
            df_male_stroke = df_stroke[df_stroke["gender"]==1]
        # Fit a logistic regression model to estimate propensity scores
        propensity_model = LogisticRegression()
        propensity_model.fit(data[covariates], data['disease'])
        propensity_scores = propensity_model.predict_proba(data[covariates])[:, 1]
        data['propensity_scores'] = propensity_scores

        # Create a DataFrame for diseaseed and control groups
        disease_group = data[data['disease'] == 1]
        control_group = data[data['disease'] == 0]

        # Fit a Nearest Neighbors model on the control group
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(control_group[covariates])

        # Find the nearest neighbors for each diseaseed unit
        distances, indices = knn.kneighbors(disease_group[covariates])
        matched_pairs = pd.DataFrame({
            'disease_index': disease_group.index,
            'control_index': control_group.index[indices.flatten()],
            'distance': distances.flatten(),
            'propensity_score': propensity_scores[indices.flatten()]
        })
        # print("===== Done! =====")
        # embed(globals(), locals())
        # Use the matched pairs to create the matched data
        control_samples = control_group.iloc[indices.flatten()]            
        print(matched_pairs)
        print(stroke_cohort)
        print(gender)
        print(control_samples)
        
##############################################################################
##############################################################################
        X = covariates
        y = target
        if gender == "Female":
            df_female = control_samples.copy()
            df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
            df_female_stroke = predict_hgs(df_female_stroke, X, y, female_best_model_trained, target)
            corr_female_control = spearmanr(df_female[f"{target}_predicted"], df_female[f"{target}_actual"])[0]
            corr_female_stroke = spearmanr(df_female_stroke[f"{target}_predicted"], df_female_stroke[f"{target}_actual"])[0]

        elif gender == "Male":
            df_male = control_samples.copy()
            df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)
            df_male_stroke = predict_hgs(df_male_stroke, X, y, male_best_model_trained, target)
            corr_male_control = spearmanr(df_male[f"{target}_predicted"], df_male[f"{target}_actual"])[0]
            corr_male_stroke = spearmanr(df_male_stroke[f"{target}_predicted"], df_male_stroke[f"{target}_actual"])[0]

    df_both_gender = pd.concat([df_female, df_male], axis=0)
    df_both_stroke = pd.concat([df_female_stroke, df_male_stroke], axis=0)
    corr_control = spearmanr(df_both_gender[f"{target}_predicted"], df_both_gender[f"{target}_actual"])[0]
    corr_stroke = spearmanr(df_both_stroke[f"{target}_predicted"], df_both_stroke[f"{target}_actual"])[0]
    text_control = 'r= ' + str(format(corr_control, '.3f'))
    text_stroke = 'r= ' + str(format(corr_stroke, '.3f'))
    text_control_female = 'r= ' + str(format(corr_female_control, '.3f'))
    text_stroke_female = 'r= ' + str(format(corr_female_stroke, '.3f'))
    text_control_male = 'r= ' + str(format(corr_male_control, '.3f'))
    text_stroke_male = 'r= ' + str(format(corr_male_stroke, '.3f'))
    # print("===== Done! =====")
    # embed(globals(), locals())
    print(df_both_gender)
    print(df_both_stroke)
##############################################################################
##############################################################################
    print(axj)
    sns.regplot(data=df_both_gender, x=f"{target}_actual", y=f"{target}_predicted", scatter=False, ax=ax[0][axj])
    sns.scatterplot(data=df_both_gender, x=f"{target}_actual", y=f"{target}_predicted", hue="gender", ax=ax[0][axj], palette = custom_palette)

    sns.regplot(data=df_both_stroke, x=f"{target}_actual", y=f"{target}_predicted", scatter=False, ax=ax[1][axj])
    sns.scatterplot(data=df_both_stroke, x=f"{target}_actual", y=f"{target}_predicted", hue="gender", ax=ax[1][axj], palette = custom_palette)
    
    legend1 = ax[0][axj].legend()
    legend2 = ax[1][axj].legend()
    # Remove the legend from the axis
    legend1.remove()
    legend2.remove()
    # Define the legend labels for each axis
    legend_labels = [f"Female(N={len(df_female)})", f"Male(N={len(df_male)})"]
    # Set a single y-label for all axes without using g
    for i in range(0,2):
        ax[i][axj].set_ylabel("Predicted HGS", fontsize=12, fontweight="bold")
        # xmin, xmax = ax[i][axj].get_xlim()
        # ymin, ymax = ax[i][axj].get_ylim()
        # if ((xmin < xmin_main) | (xmax > xmax_main)):
        #     xmin_main = xmin
        #     xmax_main = xmax
        # if ((ymin < ymin_main) | (ymax > ymax_main)):
        #     ymin_main = ymin
        #     ymax_main = ymax
        if i == 0:
            ax[i][axj].set_xlabel("")
        else:
            ax[i][axj].set_xlabel("True HGS", fontsize=12, fontweight="bold")

    if axj == 0:
        ax[0][axj].set_title(f"Control Samples - {text_control}\n(female_corr:{text_control_female}, male_corr:{text_control_male})", fontsize=10, fontweight="bold")
        ax[1][axj].set_title(f"Pre-stroke - {text_stroke}\n(female_corr:{text_stroke_female}, male_corr:{text_stroke_male})", fontsize=10, fontweight="bold")
    elif axj == 1:
        ax[0][axj].set_title(f"Control Samples - {text_control}\n(female_corr:{text_control_female}, male_corr:{text_control_male})", fontsize=10, fontweight="bold")
        ax[1][axj].set_title(f"Post-stroke - {text_stroke}\n(female_corr:{text_stroke_female}, male_corr:{text_stroke_male})", fontsize=10, fontweight="bold")
    
    # for i in range(0,2):
        # ax[i][axj].set_xlim(xmin_main, xmax_main)
        # ax[i][axj].set_ylim(ymin_main, ymax_main)
        # ax[i][axj].plot([xmin_main, xmax_main], [ymin_main, ymax_main], 'k--')
        
    # for i in range(0,2):
    #     xmin, xmax = ax[i][axj].get_xlim()
    #     ymin, ymax = ax[i][axj].get_ylim()
        # if axj==0:
        #     ax[i][0].text(xmin - 0.05 * xmax, ymax - 0.01 * ymin, text_control, verticalalignment='bottom',
        #             horizontalalignment='right', fontsize=12, fontweight="bold")
        #     ax[i][0].text(xmin - 0.05 * xmax, ymax - 0.01 * ymin, text_stroke, verticalalignment='bottom',
        #             horizontalalignment='right', fontsize=12, fontweight="bold")
        # if axj==1:
        #     ax[i][1].text(xmin - 0.05 * xmax, ymax - 0.01 * ymin, text_control, verticalalignment='bottom',
        #             horizontalalignment='right', fontsize=12, fontweight="bold")
        #     ax[i][1].text(xmin - 0.05 * xmax, ymax - 0.01 * ymin, text_stroke, verticalalignment='bottom',
        #             horizontalalignment='right', fontsize=12, fontweight="bold")
            
plt.suptitle(f"Control smaples vs patients - {target_label}", fontsize=14, fontweight="bold")
plt.show()
plt.savefig(f"control_samples_predicted_{target}.png")
plt.close()


print("===== Done! =====")
embed(globals(), locals())