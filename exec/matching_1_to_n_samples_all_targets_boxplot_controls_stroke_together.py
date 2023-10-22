import math
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
from scipy.stats import zscore
import matplotlib.patheffects as path_effects


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

###############################################################################
for target in ["hgs_L+R", "hgs_left", "hgs_right"]:
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

    stroke_cohort = "longitudinal-stroke"
    session_column = f"1st_{stroke_cohort}_session"
    df_stroke = stroke.load_hgs_predicted_results("stroke", mri_status, session_column, model_name, feature_type, target, "both_gender")
    df_stroke.loc[:, "disease"] = 1
    df_stroke = df_stroke.drop(index=1872273)

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
    X = covariates
    y = target
    custom_palette = sns.color_palette(['#a851ab', '#005c95'])  # You can use any hex color codes you prefer
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    for stroke_cohort in ["pre-stroke", "post-stroke"]:
        if stroke_cohort == "pre-stroke":
            df = df_pre.copy()
            df_stroke = df_pre[df_pre['disease']==1]
            axj=0
            print(stroke_cohort)
        elif stroke_cohort == "post-stroke":
            df = df_post.copy()
            df_stroke = df_post[df_post['disease']==1]
            axj=1
    ##############################################################################
        control_samples_female = pd.DataFrame()
        control_samples_male = pd.DataFrame()
        for gender in ["Female", "Male"]:
            if gender == "Female":
                data = df[df["gender"]==0]
                df_female_stroke = df_stroke[df_stroke["gender"]==0]
                print(gender)
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

            matched_pairs = pd.DataFrame({'disease_index': disease_group.index})
            matched_data = pd.DataFrame()
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
                # matched_data = matched_data.append(matched_data_tmp)
                matched_data = pd.concat([matched_data, matched_data_tmp], axis=0)
                if gender == "Female":
                    control_samples_female = pd.concat([control_samples_female, control_group.iloc[indices[:,k-1].flatten()]], axis=0)
                    df_female = control_samples_female.copy()
                    df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
                    df_female_stroke = predict_hgs(df_female_stroke, X, y, female_best_model_trained, target)
                    corr_female_control = spearmanr(df_female[f"{target}_predicted"], df_female[f"{target}_actual"])[0]
                    corr_female_stroke = spearmanr(df_female_stroke[f"{target}_predicted"], df_female_stroke[f"{target}_actual"])[0]
                elif gender == "Male":                
                    control_samples_male = pd.concat([control_samples_male, control_group.iloc[indices[:,k-1].flatten()]], axis=0)
                    df_male = control_samples_male.copy()
                    df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)
                    df_male_stroke = predict_hgs(df_male_stroke, X, y, male_best_model_trained, target)
                    corr_male_control = spearmanr(df_male[f"{target}_predicted"], df_male[f"{target}_actual"])[0]
                    corr_male_stroke = spearmanr(df_male_stroke[f"{target}_predicted"], df_male_stroke[f"{target}_actual"])[0]
            print(matched_data)
            # matched_patients[f'propensity_scores_disease_{k}'] = matched_data[k-1]['propensity_scores_disease']
            # matched_controls[f'propensity_scores_disease_{k}'] = matched_data[k-1]['propensity_scores_control']
            # unmatched_controls[f'propensity_scores_disease_{k}'] = control_group[~control_group.index.isin(matched_data[k-1]['index_control'])].loc[:, 'propensity_scores']
            # unmatched_patients[f'propensity_scores_disease_{k}'] = disease_group[~disease_group.index.isin(matched_data[k-1]['index_disease'])].loc[:, 'propensity_scores']
            print(matched_pairs)
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
        print(df_both_gender)
        print(df_both_stroke)
        df_both_gender = df_both_gender.drop(columns=f"{target}")
        df_both_gender.rename(columns={f'{target}_actual':"actual", f"{target}_predicted":"predicted", f"{target}_(actual-predicted)": "delta"}, inplace=True)
        if target == "hgs_L+R":
            if stroke_cohort == "pre-stroke":
                df_l_r_pre = df_both_gender
                df_l_r_pre['hgs_target'] = "HGS L+R"
            elif stroke_cohort == "post-stroke":
                df_l_r_post = df_both_gender
                df_l_r_post['hgs_target'] = "HGS L+R"
        elif target == "hgs_left":
            if stroke_cohort == "pre-stroke":
                df_left_pre = df_both_gender
                df_left_pre['hgs_target'] = "HGS Left"
            elif stroke_cohort == "post-stroke":
                df_left_post = df_both_gender
                df_left_post['hgs_target'] = "HGS Left"
        elif target == "hgs_right":
            if stroke_cohort == "pre-stroke":
                df_right_pre = df_both_gender
                df_right_pre['hgs_target'] = "HGS Right"
            elif stroke_cohort == "post-stroke":
                df_right_post = df_both_gender
                df_right_post['hgs_target'] = "HGS Right"

    ##############################################################################
    ##############################################################################
    
df_both_pre = pd.concat([df_left_pre, df_right_pre, df_l_r_pre])
df_both_pre['stroke_cohort'] = "pre"
df_both_post = pd.concat([df_left_post, df_right_post, df_l_r_post])
df_both_post['stroke_cohort'] = "post"

df = pd.concat([df_both_pre, df_both_post])
###############################################################################
def add_median_labels(ax, fmt='.3f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=10)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

###############################################################################
# Define a custom palette with two blue colors
custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
sns.set(style="whitegrid")
ax = sns.boxplot(x="hgs_target", y="delta", hue="stroke_cohort", hue_order=["pre", "post"], data=df, palette=custom_palette)    
# Add labels and title
plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
plt.ylabel("HGS delta values", fontsize=20, fontweight="bold")
plt.title(f"Matching samples from controls HGS delta values", fontsize=15, fontweight="bold")

# Show the plot
legend = plt.legend(title="Macthing samples cohort", loc="upper left", prop={'size': 8})  # Add legend
# Modify individual legend labels
legend.get_texts()[0].set_text(f"Matching controls of Pre-stroke: N={len(df_both_gender)}")
legend.get_texts()[1].set_text(f"Matching controls of Post-stroke: N={len(df_both_gender)}")

plt.tight_layout()

add_median_labels(ax)

plt.show()
plt.savefig(f"boxplot_samples_{population}_{feature_type}_hgs_both_gender.png")
plt.close()
###############################################################################
melted_df_tmp = pd.melt(df, id_vars=['hgs_target', 'stroke_cohort', 'gender'], var_name='variable', value_name='value')
melted_df = melted_df_tmp[melted_df_tmp['variable']=='delta']
melted_df['combine_hgs_stroke_cohort_category'] = melted_df['hgs_target'] + '-' + melted_df['stroke_cohort']
custom_palette = sns.color_palette(['#a851ab', '#005c95'])  # You can use any hex color codes you prefer

plt.figure(figsize=(20, 8))  # Adjust the figure size if needed
sns.set(style="whitegrid")
ax = sns.boxplot(x="combine_hgs_stroke_cohort_category", y="value", hue="gender", data=melted_df, palette=custom_palette)    
# Add labels and title
plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
plt.ylabel("HGS delta values", fontsize=20, fontweight="bold")
plt.title(f"Matching samples from controls HGS delta values", fontsize=15, fontweight="bold")

# Show the plot
legend = plt.legend(title="Macthing samples cohort", loc="upper left", prop={'size': 8})  # Add legend
# Modify individual legend labels
legend.get_texts()[0].set_text(f"Matching controls Female: N={len(df_both_gender[df_both_gender['gender']==0])}")
legend.get_texts()[1].set_text(f"Matching controls Male: N={len(df_both_gender[df_both_gender['gender']==1])}")

plt.tight_layout()

add_median_labels(ax)

plt.show()
plt.savefig(f"boxplot_samples_{population}_{feature_type}_hgs_separate_gender.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())