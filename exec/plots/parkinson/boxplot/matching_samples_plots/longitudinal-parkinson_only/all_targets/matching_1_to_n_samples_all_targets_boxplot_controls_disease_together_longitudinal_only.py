import math
import sys
import numpy as np
import pandas as pd
import os
from psmpy import PsmPy
from hgsprediction.load_results import healthy
from hgsprediction.load_data import healthy_load_data
from hgsprediction.load_results import parkinson
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import predict_hgs
from scipy.stats import spearmanr
from scipy.stats import zscore
import matplotlib.patheffects as path_effects
from scipy.stats import ranksums

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
pd.options.mode.chained_assignment = None  # 'None' suppresses the warning

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
 
    ###############################################################################

    parkinson_cohort = "longitudinal-parkinson"
    session_column = f"1st_{parkinson_cohort}_session"
    df_parkinson = parkinson.load_hgs_predicted_results("parkinson", mri_status, session_column, model_name, feature_type, target, "both_gender")
    df_parkinson.loc[:, "disease"] = 1
    print("===== Done! =====")
    embed(globals(), locals())
    # df_parkinson = df_parkinson.drop(index=1872273)

    df_pre_parkinson = df_parkinson.loc[:, ["gender", "1st_pre-parkinson_age", "1st_pre-parkinson_bmi",  "1st_pre-parkinson_height",  "1st_pre-parkinson_waist_to_hip_ratio", f"1st_pre-parkinson_{target}", "disease"]]
    df_pre_parkinson.rename(columns={"1st_pre-parkinson_age":"age", "1st_pre-parkinson_bmi":"bmi",  "1st_pre-parkinson_height":"height",  "1st_pre-parkinson_waist_to_hip_ratio":"waist_to_hip_ratio", 
                                "1st_pre-parkinson_handedness":"handedness", f"1st_pre-parkinson_{target}":f"{target}"}, inplace=True)

    df_post_parkinson = df_parkinson.loc[:, ["gender", "1st_post-parkinson_age", "1st_post-parkinson_bmi",  "1st_post-parkinson_height",  "1st_post-parkinson_waist_to_hip_ratio", f"1st_post-parkinson_{target}", "disease"]]
    df_post_parkinson.rename(columns={"1st_post-parkinson_age":"age", "1st_post-parkinson_bmi":"bmi",  "1st_post-parkinson_height":"height",  "1st_post-parkinson_waist_to_hip_ratio":"waist_to_hip_ratio",
                                "1st_post-parkinson_handedness":"handedness", f"1st_post-parkinson_{target}":f"{target}"}, inplace=True)
    ###############################################################################
    df_pre = pd.concat([df_healthy, df_pre_parkinson], axis=0)
    df_pre.insert(0, "index", df_pre.index)

    df_post = pd.concat([df_healthy, df_post_parkinson], axis=0)
    df_post.insert(0, "index", df_post.index)

    ##############################################################################
    # Define the covariates you want to use for matching
    # covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]
    covariates = ["bmi", "height", "waist_to_hip_ratio", "age"]
    X = covariates
    y = target
    custom_palette = sns.color_palette(['#a851ab', '#005c95'])  # You can use any hex color codes you prefer
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    for parkinson_cohort in ["pre-parkinson", "post-parkinson"]:
        if parkinson_cohort == "pre-parkinson":
            df = df_pre.copy()
            df_parkinson = df_pre[df_pre['disease']==1]
        elif parkinson_cohort == "post-parkinson":
            df = df_post.copy()
            df_parkinson = df_post[df_post['disease']==1]
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
            matched_patients = pd.DataFrame()
            matched_controls = pd.DataFrame()
            unmatched_controls = pd.DataFrame()
            unmatched_patients = pd.DataFrame()
            # Define the range of k from 1 to n
            n = 1  # You can change this to the desired value of n
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
        df_both_gender = df_both_gender.drop(columns=f"{target}")
        df_both_gender.rename(columns={f'{target}_actual':"actual", f"{target}_predicted":"predicted", f"{target}_(actual-predicted)": "delta"}, inplace=True)
        df_both_parkinson = df_both_parkinson.drop(columns=f"{target}")
        df_both_parkinson.rename(columns={f'{target}_actual':"actual", f"{target}_predicted":"predicted", f"{target}_(actual-predicted)": "delta"}, inplace=True)
        if target == "hgs_L+R":
            if parkinson_cohort == "pre-parkinson":
                df_l_r_pre = df_both_gender
                df_l_r_pre['hgs_target'] = "HGS L+R"
                df_l_r_parkinson_pre = df_both_parkinson                
                df_l_r_parkinson_pre['hgs_target'] = "HGS L+R"                
            elif parkinson_cohort == "post-parkinson":
                df_l_r_post = df_both_gender
                df_l_r_post['hgs_target'] = "HGS L+R"
                df_l_r_parkinson_post = df_both_parkinson                
                df_l_r_parkinson_post['hgs_target'] = "HGS L+R"                 
        elif target == "hgs_left":
            if parkinson_cohort == "pre-parkinson":
                df_left_pre = df_both_gender
                df_left_pre['hgs_target'] = "HGS Left"
                df_left_parkinson_pre = df_both_parkinson                
                df_left_parkinson_pre['hgs_target'] = "HGS Left"                 
            elif parkinson_cohort == "post-parkinson":
                df_left_post = df_both_gender
                df_left_post['hgs_target'] = "HGS Left"
                df_left_parkinson_post = df_both_parkinson                
                df_left_parkinson_post['hgs_target'] = "HGS Left"                  
        elif target == "hgs_right":
            if parkinson_cohort == "pre-parkinson":
                df_right_pre = df_both_gender
                df_right_pre['hgs_target'] = "HGS Right"
                df_right_parkinson_pre = df_both_parkinson                
                df_right_parkinson_pre['hgs_target'] = "HGS Right"                  
            elif parkinson_cohort == "post-parkinson":
                df_right_post = df_both_gender
                df_right_post['hgs_target'] = "HGS Right"
                df_right_parkinson_post = df_both_parkinson                
                df_right_parkinson_post['hgs_target'] = "HGS Right"    

    ##############################################################################
    ##############################################################################
    
df_both_pre = pd.concat([df_left_pre, df_right_pre, df_l_r_pre])
df_both_pre['parkinson_cohort'] = "pre"
df_both_post = pd.concat([df_left_post, df_right_post, df_l_r_post])
df_both_post['parkinson_cohort'] = "post"

df = pd.concat([df_both_pre, df_both_post])

df_both_parkinson_pre = pd.concat([df_left_parkinson_pre, df_right_parkinson_pre, df_l_r_parkinson_pre])
df_both_parkinson_pre['parkinson_cohort'] = "pre"
df_both_parkinson_post = pd.concat([df_left_parkinson_post, df_right_parkinson_post, df_l_r_parkinson_post])
df_both_parkinson_post['parkinson_cohort'] = "post"

df_parkinson_together = pd.concat([df_both_parkinson_pre, df_both_parkinson_post])

###############################################################################
def add_median_labels(ax, fmt='.3f'):
    xticks_positios_array = []
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=12)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array
###############################################################################
###############################################################################
# print("===== Done! =====")
# embed(globals(), locals())
df["hgs_target_parkinson_cohort"] = df["hgs_target"] + "-" +df["parkinson_cohort"]
df_parkinson_together["hgs_target_parkinson_cohort"] = df_parkinson_together["hgs_target"] + "-" +df_parkinson_together["parkinson_cohort"]
df_main = pd.concat([df, df_parkinson_together])
for y_axis in ["actual", "predicted", "delta"]:
    melted_df = pd.melt(df_main, id_vars=["hgs_target_parkinson_cohort", "disease"], value_vars=y_axis, var_name="variable", ignore_index=False)
    # Initialize a list to store the test results
    results = pd.DataFrame(columns=["hgs_target_parkinson_cohort", "ranksum_stat", "ranksum_p_value", f"max_sample_{y_axis}", f"max_parkinson_{y_axis}"])
    for i, hgs_target_parkinson_cohort in enumerate(["HGS Left-pre", "HGS Left-post", "HGS Right-pre", "HGS Right-post", "HGS L+R-pre", "HGS L+R-post"]):
        tmp = melted_df[melted_df["hgs_target_parkinson_cohort"]== hgs_target_parkinson_cohort]
        tmp_samples = tmp[tmp["disease"]==0]
        tmp_parkinson = tmp[tmp["disease"]==1]
        stat, p_value = ranksums(tmp_samples["value"], tmp_parkinson["value"])
        print(tmp)
        print(stat, p_value)
        results.loc[i, "hgs_target_parkinson_cohort"] = hgs_target_parkinson_cohort
        results.loc[i, "ranksum_stat"] = stat
        results.loc[i, "ranksum_p_value"] = p_value
        results.loc[i, f"max_sample_{y_axis}"] = tmp_samples["value"].max()
        results.loc[i, f"max_parkinson_{y_axis}"] = tmp_parkinson["value"].max()

    # Define a custom palette with two blue colors
    custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer
    plt.figure(figsize=(18, 10))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    # Define the order in which you want the x-axis categories
    x_order = ['HGS Left-pre', 'HGS Left-post', 'HGS Right-pre', 'HGS Right-post', 'HGS L+R-pre', 'HGS L+R-post']
    ax = sns.boxplot(data=melted_df, x="hgs_target_parkinson_cohort", y="value", hue="disease", order=x_order, palette=custom_palette)   
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"HGS {y_axis.capitalize()} values", fontsize=20, fontweight="bold")
    plt.title(f"Matching samples from controls vs parkinson HGS {y_axis.capitalize()} values", fontsize=15, fontweight="bold")

    ymin, ymax = plt.ylim()
    plt.yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10), fontsize=18, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    legend = plt.legend(loc="upper left", prop={'size': 16, 'weight': 'bold'})
    legend.set_title("Samples", {'size': 16, 'weight': 'bold'})
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"Matching samples from controls(N={len(df_both_gender)})")
    legend.get_texts()[1].set_text(f"parkinson(N={len(df_parkinson)})")

    plt.tight_layout()

    xticks_positios_array = add_median_labels(ax)

    for i, x_box_pos in enumerate(np.arange(0,11,2)):
        x1 = xticks_positios_array[x_box_pos]
        x2 = xticks_positios_array[x_box_pos+1]
        y, h, col = results.loc[i, f"max_sample_{y_axis}"] + 2, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f"p={results.loc[i, 'ranksum_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold',  color=col)

    plt.show()
    plt.savefig(f"boxplot_1_to_{n}_samples_{session_column}_{y_axis}_{population}_{feature_type}_hgs_both_gender_controls_parkinson.png")
    plt.close()
###############################################################################

    melted_df_female = pd.melt(df_main[df_main["gender"]==0], id_vars=["hgs_target_parkinson_cohort", "disease"], value_vars=y_axis, var_name="variable", ignore_index=False)
    results_female = pd.DataFrame(columns=["hgs_target_parkinson_cohort", "ranksum_stat", "ranksum_p_value", f"max_sample_{y_axis}", f"max_parkinson_{y_axis}"])
    for i, hgs_target_parkinson_cohort in enumerate(["HGS Left-pre", "HGS Left-post", "HGS Right-pre", "HGS Right-post", "HGS L+R-pre", "HGS L+R-post"]):
        tmp = melted_df_female[melted_df_female["hgs_target_parkinson_cohort"]== hgs_target_parkinson_cohort]
        tmp_samples = tmp[tmp["disease"]==0]
        tmp_parkinson = tmp[tmp["disease"]==1]
        stat, p_value = ranksums(tmp_samples["value"], tmp_parkinson["value"])
        print(tmp)
        print(stat, p_value)
        results_female.loc[i, "hgs_target_parkinson_cohort"] = hgs_target_parkinson_cohort
        results_female.loc[i, "ranksum_stat"] = stat
        results_female.loc[i, "ranksum_p_value"] = p_value
        results_female.loc[i, f"max_sample_{y_axis}"] = tmp_samples["value"].max()
        results_female.loc[i, f"max_parkinson_{y_axis}"] = tmp_parkinson["value"].max()
    custom_palette = sns.color_palette(['#ca96cc', '#a851ab'])  # You can use any hex color codes you prefer
    plt.figure(figsize=(18, 10))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    x_order = ['HGS Left-pre', 'HGS Left-post', 'HGS Right-pre', 'HGS Right-post', 'HGS L+R-pre', 'HGS L+R-post']
    ax = sns.boxplot(data=melted_df_female, x="hgs_target_parkinson_cohort", y="value", hue="disease", order=x_order, palette=custom_palette)    
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"HGS {y_axis.capitalize()} values", fontsize=20, fontweight="bold")
    plt.title(f"Matching samples from controls vs parkinson HGS {y_axis.capitalize()} values - Females", fontsize=15, fontweight="bold")

    ymin, ymax = plt.ylim()
    plt.yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10), fontsize=18, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    
    legend = plt.legend(loc="upper left", prop={'size': 16, 'weight': 'bold'})
    legend.set_title("Samples", {'size': 16, 'weight': 'bold'})    # Modify individual legend labels
    female_matching_samples_number = len(df_both_gender[df_both_gender["gender"]==0])
    female_parkinson_number = len(df_both_parkinson[df_both_parkinson["gender"]==0])
    legend.get_texts()[0].set_text(f"Matching samples from controls Female(N={female_matching_samples_number})")
    legend.get_texts()[1].set_text(f"parkinson Female(N={female_parkinson_number})")

    plt.tight_layout()

    xticks_positios_array = add_median_labels(ax)

    for i, x_box_pos in enumerate(np.arange(0,11,2)):
        x1 = xticks_positios_array[x_box_pos]
        x2 = xticks_positios_array[x_box_pos+1]
        y, h, col = results_female.loc[i, f"max_sample_{y_axis}"] + 2, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f"p={results_female.loc[i, 'ranksum_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold', color=col)

    plt.show()
    plt.savefig(f"boxplot_1_to_{n}_samples_{session_column}_{y_axis}_{population}_{feature_type}_hgs_separate_gender_separated_parkinson_Female.png")
    plt.close()

###############################################################################

    melted_df_male = pd.melt(df_main[df_main["gender"]==1], id_vars=["hgs_target_parkinson_cohort", "disease"], value_vars=y_axis, var_name="variable", ignore_index=False)
    results_male = pd.DataFrame(columns=["hgs_target_parkinson_cohort", "ranksum_stat", "ranksum_p_value", f"max_sample_{y_axis}", f"max_parkinson_{y_axis}"])
    for i, hgs_target_parkinson_cohort in enumerate(["HGS Left-pre", "HGS Left-post", "HGS Right-pre", "HGS Right-post", "HGS L+R-pre", "HGS L+R-post"]):
        tmp = melted_df_male[melted_df_male["hgs_target_parkinson_cohort"]== hgs_target_parkinson_cohort]
        tmp_samples = tmp[tmp["disease"]==0]
        tmp_parkinson = tmp[tmp["disease"]==1]
        stat, p_value = ranksums(tmp_samples["value"], tmp_parkinson["value"])
        print(tmp)
        print(stat, p_value)
        results_male.loc[i, "hgs_target_parkinson_cohort"] = hgs_target_parkinson_cohort
        results_male.loc[i, "ranksum_stat"] = stat
        results_male.loc[i, "ranksum_p_value"] = p_value
        results_male.loc[i, f"max_sample_{y_axis}"] = tmp_samples["value"].max()
        results_male.loc[i, f"max_parkinson_{y_axis}"] = tmp_parkinson["value"].max()

    custom_palette = sns.color_palette(['#669dbf', '#005c95'])  # You can use any hex color codes you prefer
    plt.figure(figsize=(18, 10))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    x_order = ['HGS Left-pre', 'HGS Left-post', 'HGS Right-pre', 'HGS Right-post', 'HGS L+R-pre', 'HGS L+R-post']
    ax = sns.boxplot(data=melted_df_male, x="hgs_target_parkinson_cohort", y="value", hue="disease", order=x_order, palette=custom_palette)    
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"HGS {y_axis.capitalize()} values", fontsize=20, fontweight="bold")
    plt.title(f"Matching samples from controls vs parkinson HGS {y_axis.capitalize()} values - Males", fontsize=15, fontweight="bold")

    ymin, ymax = plt.ylim()
    plt.yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10), fontsize=18, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    legend = plt.legend(loc="upper left", prop={'size': 16, 'weight': 'bold'})
    legend.set_title("Samples", {'size': 16, 'weight': 'bold'})
    male_matching_samples_number = len(df_both_gender[df_both_gender["gender"]==1])
    male_parkinson_number = len(df_both_parkinson[df_both_parkinson["gender"]==1])
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"Matching samples from controls Male(N={male_matching_samples_number})")
    legend.get_texts()[1].set_text(f"parkinson Male(N={male_parkinson_number})")

    plt.tight_layout()

    xticks_positios_array = add_median_labels(ax)

    for i, x_box_pos in enumerate(np.arange(0,11,2)):
        x1 = xticks_positios_array[x_box_pos]
        x2 = xticks_positios_array[x_box_pos+1]
        y, h, col = results_male.loc[i, f"max_sample_{y_axis}"] + 2, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f"p={results_male.loc[i, 'ranksum_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold', color=col)

    plt.show()
    plt.savefig(f"boxplot_1_to_{n}_samples_{session_column}_{y_axis}_{population}_{feature_type}_hgs_separate_gender_separated_parkinson_Male.png")
    plt.close()


print("===== Done! =====")
embed(globals(), locals())