
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
covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio"]

for stroke_cohort in ["pre-stroke", "post-stroke"]:
    if stroke_cohort == "pre-stroke":
        df = df_pre.copy()
        print(stroke_cohort)
    elif stroke_cohort == "post-stroke":
        df = df_post.copy()
##############################################################################
    for gender in ["Female", "Male"]:
        if gender == "Female":
            data = df[df["gender"]==0]
            # print("===== Done! =====")
            # embed(globals(), locals())
        elif gender == "Male":    
            data = df[df["gender"]==1]
            axi = 1
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
        
        # Use the matched pairs to create the matched data
        matched_data = disease_group.reset_index(drop=True).join(control_group.iloc[indices.flatten()].reset_index(drop=True), lsuffix="_disease", rsuffix="_control")

        matched_data['distance'] = matched_pairs['distance'].values

        matched_patients = matched_data['propensity_scores_disease']
        matched_controls = matched_data['propensity_scores_control']
        unmatched_controls= control_group[~control_group.index.isin(pd.concat([matched_data['index_disease'], matched_data['index_control']]))].loc[:, 'propensity_scores']
        unmatched_patients= disease_group[~disease_group.index.isin(pd.concat([matched_data['index_disease'], matched_data['index_control']]))].loc[:, 'propensity_scores']
        print(len(matched_patients))
        print(len(matched_controls))
        print(stroke_cohort)
        print(gender)
        print(matched_data)
        print(matched_data.describe())
        print("===== Done! =====")
        embed(globals(), locals())
        matched_data.to_csv(f"{stroke_cohort}_{gender}_{target}_with_hgs", sep=',', index=False)
##############################################################################
##############################################################################
        # plot
        fig, ax = plt.subplots(2,2, figsize=(18, 10))
        # Set the background color of the axes to white
        sns.set_style('darkgrid')
        for i, feature in enumerate(covariates):
            if i == 0:
                axi = 0
                axj = 0
            elif i == 1:
                axi = 0
                axj = 1
            elif i == 2:
                axi = 1
                axj = 0
            elif i == 3:
                axi = 1
                axj = 1
            y1 = np.full((len(matched_data)), 0.6)
            ax[axi][axj].scatter(matched_data[feature+"_disease"], y1, s=100, edgecolors='black', c='white')
            df_y1 = pd.concat([matched_data[feature+"_disease"], pd.Series(y1, name='yaxis')], axis=1)
            df_y1 = df_y1.rename(columns={f"{feature}_disease":f"{feature}", f"{feature}_control":f"{feature}"})
            y2 = np.full((len(matched_data)), 0.2)
            ax[axi][axj].scatter(matched_data[feature+"_control"], y2, s=100, edgecolors='black', c='white')
            df_y2 = pd.concat([matched_data[feature+"_control"], pd.Series(y2, name='yaxis')], axis=1)
            df_y2 = df_y2.rename(columns={f"{feature}_disease":f"{feature}", f"{feature}_control":f"{feature}"})
            
            # Plot lines connecting corresponding points
            ax[axi][axj].plot([df_y1[feature], df_y2[feature]], [df_y1["yaxis"], df_y2["yaxis"]], linestyle='-')
            xmin, xmax = ax[axi][axj].get_xlim()
            x_axis_mean = (xmin + xmax) / 2
            ax[axi][axj].set_ylim(0, .8)

            text1 = f"Matched Disease Patients(N={len(matched_patients)})"
            text2 = f"Matched Healthy Controls(N={len(matched_controls)})"

            ax[axi][axj].text(x_axis_mean, 0.6 + 0.08, text1, horizontalalignment='center', fontsize=12, fontweight="bold")
            ax[axi][axj].text(x_axis_mean, 0.2 - 0.08, text2, horizontalalignment='center', fontsize=12, fontweight="bold")
            # Remove the y-axis ticks
            ax[axi][axj].set_yticklabels([])
            ax[axi][axj].set_title(f"{feature.capitalize()}", fontsize=14, fontweight="bold")
    
        fig.suptitle(f"Distribution of Features\n{stroke_cohort}\nTarget={target_label}\n{gender}(Total Controls={len(pd.concat([matched_controls, unmatched_controls], axis=0))})", fontsize=16, fontweight="bold")
        plt.show()
        plt.savefig(f"FT_distribution_{target}_{stroke_cohort}_{gender}_without_hgs.png")
        plt.close()

##############################################################################
##############################################################################

# a = pd.read_csv(f"{stroke_cohort}_{gender}_{target}", sep=',')
print("===== Done! =====")
embed(globals(), locals())