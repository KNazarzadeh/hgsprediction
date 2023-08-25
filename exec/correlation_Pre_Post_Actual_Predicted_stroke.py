
import pandas as pd
import numpy as np
import pickle
import os
from hgsprediction.input_arguments import parse_args, input_arguments
from LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
# import pysurfer as ps
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model_type, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)

###############################################################################
if confound_status == 0:
        confound = "without_confound_removal"
else:
    confound = "with_confound_removal"
if model_type == "rf":
    model_name = "random_forest"
if model_type == "linear_svm":
    model_name = "linear_svm"
if "+" in feature_type:
        feature_type_name = feature_type.replace("+", "_")
else:
    feature_type_name = feature_type
if target == "L+R":
    target_label = "L_plus_R"
else:
    target_label = target
open_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type_name}_features",
        f"{target_label}_target",
        f"{model_name}",
        f"{confound}",
        f"{cv_repeats_number}_repeats_{cv_folds_number}_folds",
        f"results_csv",
        "model_trained",
    )

# Define the csv file path to save
open_file_path = os.path.join(
    open_folder_path,
    f"main_model_trained_{mri_status}_{population}_{gender}_genders_{feature_type_name}_{target_label}_{model_name}_{confound}_{cv_repeats_number}_repeats_{cv_folds_number}_folds.pkl")

with open(open_file_path, 'rb') as f:
    model_trained = pickle.load(f)

# Define the csv file path to save
open_file_path_female = os.path.join(
    open_folder_path,
    f"model_trained_female_{mri_status}_{population}_{gender}_genders_{feature_type_name}_{target_label}_{model_name}_{confound}_{cv_repeats_number}_repeats_{cv_folds_number}_folds.pkl")

with open(open_file_path_female, 'rb') as f:
    model_trained_female = pickle.load(f)
    # Define the csv file path to save
open_file_path_male = os.path.join(
    open_folder_path,
    f"model_trained_male_{mri_status}_{population}_{gender}_genders_{feature_type_name}_{target_label}_{model_name}_{confound}_{cv_repeats_number}_repeats_{cv_folds_number}_folds.pkl")

with open(open_file_path_male, 'rb') as f:
    model_trained_male = pickle.load(f)

###############################################################################
mri_status = "mri"
population = "stroke"

folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        f"data_{motor}",
        population,
        "prepared_data",
        f"{mri_status}_{population}",
    )

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"longitudinal_disease_{mri_status}_{population}.csv")
    
data_longitudinal = pd.read_csv(file_path, sep=',')

##############################################################################
data_longitudinal.loc[:, f"1_post_hgs(L+R)"] = \
            data_longitudinal.loc[:, f"1_post_left_hgs"] + data_longitudinal.loc[:, f"1_post_right_hgs"]

###############################################################################
data_longitudinal.loc[:, f"1_pre_hgs(L+R)"] = \
            data_longitudinal.loc[:, f"1_pre_left_hgs"] + data_longitudinal.loc[:, f"1_pre_right_hgs"]

##############################################################################
# Replace Age
for ses in range(0,4):
    sub_id = data_longitudinal[data_longitudinal['1_post_session']== f"session-{ses}.0"].index.values
    data_longitudinal.loc[sub_id, '1_post_age'] = data_longitudinal.loc[sub_id, f'21003-{ses}.0']
    sub_id = data_longitudinal[data_longitudinal['1_pre_session']== f"session-{ses}.0"].index.values
    data_longitudinal.loc[sub_id, '1_pre_age'] = data_longitudinal.loc[sub_id, f'21003-{ses}.0']
##############################################################################
# Replace Height
for ses in range(0,4):
    sub_id = data_longitudinal[data_longitudinal['1_post_session']== f"session-{ses}.0"].index.values
    data_longitudinal.loc[sub_id, '1_post_height'] = data_longitudinal.loc[sub_id, f'50-{ses}.0']
    sub_id = data_longitudinal[data_longitudinal['1_pre_session']== f"session-{ses}.0"].index.values
    data_longitudinal.loc[sub_id, '1_pre_height'] = data_longitudinal.loc[sub_id, f'50-{ses}.0']
##############################################################################
# Replace waist to hip ratio
for ses in range(0,4):
    sub_id = data_longitudinal[data_longitudinal['1_post_session']== f"session-{ses}.0"].index.values
    data_longitudinal.loc[sub_id, '1_post_waist'] = data_longitudinal.loc[sub_id, f'48-{ses}.0']
    data_longitudinal.loc[sub_id, '1_post_hip'] = data_longitudinal.loc[sub_id, f'49-{ses}.0']
    data_longitudinal['1_post_waist_hip_ratio'] = (data_longitudinal.loc[:, "1_post_waist"].astype(str).astype(float)).div(
                data_longitudinal.loc[:, "1_post_hip"].astype(str).astype(float))
    sub_id = data_longitudinal[data_longitudinal['1_pre_session']== f"session-{ses}.0"].index.values
    data_longitudinal.loc[sub_id, '1_pre_waist'] = data_longitudinal.loc[sub_id, f'48-{ses}.0']
    data_longitudinal.loc[sub_id, '1_pre_hip'] = data_longitudinal.loc[sub_id, f'49-{ses}.0']
    data_longitudinal['1_pre_waist_hip_ratio'] = (data_longitudinal.loc[:, "1_pre_waist"].astype(str).astype(float)).div(
                data_longitudinal.loc[:, "1_pre_hip"].astype(str).astype(float))

###############################################################################
def predict_hgs(
    df,
    stroke_status,
):
    X = [f"1_{stroke_status}_age", f"1_{stroke_status}_bmi", f"1_{stroke_status}_height", f"1_{stroke_status}_waist_hip_ratio"]
    # Target: HGS(L+R)
    if target == "L+R":
        y = f"1_{stroke_status}_hgs(L+R)" 
    df = df.dropna(subset=y)
    df = df.dropna(subset=X)

    df_tmp = df[X]
    df_tmp = df_tmp.rename(columns={f"1_{stroke_status}_age": 'Age1stVisit', f'1_{stroke_status}_bmi': '21001-0.0', f'1_{stroke_status}_height':'50-0.0', f'1_{stroke_status}_waist_hip_ratio': 'WHR-0.0'})
    df_tmp = pd.concat([df_tmp, df[y],df['31-0.0']], axis=1)

    female_df = df_tmp[df_tmp['31-0.0']==0]
    male_df = df_tmp[df_tmp['31-0.0']==1]

    X = ['Age1stVisit', '21001-0.0', '50-0.0', 'WHR-0.0']
    
    return X, y, female_df, male_df

###############################################################################
def corr_calculator(
    model_trained,
    df,
    X,
    y,
):
    y_true = df[y]
    y_pred = model_trained.predict(df[X])
    df["actual_hgs"] = y_true
    df["predicted_hgs"] = y_pred
    df["hgs_diff"] = y_true - y_pred
    
    mae = format(mean_absolute_error(y_true, y_pred), '.2f')
    corr, p = spearmanr(y_pred, y_true)
    score = format(r2_score(y_true, y_pred), '.2f')
    corr_diff, p_diff = spearmanr(df["hgs_diff"], y_true)

    return df, corr, score, mae, corr_diff

###############################################################################
df_test_set = data_longitudinal.copy()
X_pre, y_pre, female_df_pre, male_df_pre = predict_hgs(df_test_set, "pre")
X_post, y_post, female_df_post, male_df_post = predict_hgs(df_test_set, "post")

male_df_pre = male_df_pre[male_df_pre.index.isin(male_df_pre.index.intersection(male_df_post.index))]
male_df_post = male_df_post[male_df_post.index.isin(male_df_post.index.intersection(male_df_pre.index))]

female_pre, corr_female_pre, score_female_pre, mae_female_pre, corr_diff_female_pre = corr_calculator(model_trained_female, female_df_pre, X_pre, y_pre)
male_pre, corr_male_pre, score_male_pre, mae_male_pre, corr_diff_male_pre = corr_calculator(model_trained_male, male_df_pre, X_pre, y_pre)
female_post, corr_female_post, score_female_post, mae_female_post, corr_diff_female_post = corr_calculator(model_trained_female, female_df_post, X_post, y_post)
male_post, corr_male_post, score_male_post, mae_male_post, corr_diff_male_post = corr_calculator(model_trained_male, male_df_post, X_post, y_post)

print("===== Done! =====")
embed(globals(), locals())
##############################################################################################################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
fig, ax = plt.subplots(2, 2, figsize=(20,20))
sns.set_context("poster")
# plt.plot(y_true_fefemale, y_pred_fefemale)
# plt.axis("equal")
ax[0,0].set_box_aspect(1)

sns.regplot(x=male_pre["actual_hgs"], y=male_pre["predicted_hgs"], ax=ax[0,0], line_kws={"color": "red"}, scatter_kws={"color": "grey"})

ax[0,0].tick_params(axis='both', labelsize=20)

xmin0, xmax0 = ax[0,0].get_xlim()
ymin0, ymax0 = ax[0,0].get_ylim()
max0 = max(xmax0, ymax0)
min0 = min(xmin0, ymin0)
ax[0,0].set_ylim(min0, max0)
ax[0,0].set_xlim(min0, max0)
xmin0, xmax0 = ax[0,0].get_xlim()
ymin0, ymax0 = ax[0,0].get_ylim()
yticks0 = ax[0,0].get_yticks()
xticks0 = ax[0,0].get_xticks()
ax[0,0].set_xticks(yticks0)
ax[0,0].set_yticks(xticks0)
text00 = 'CORR: ' + str(format(corr_male_pre, '.3f'))
ax[0,0].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
ax[0,0].set_ylabel('Predicted HGS', fontsize=20, fontweight="bold")

ax[0,0].set_title(f"Predicted vs Actual HGS - Pre-stroke - Males(N={len(male_post)})", fontsize=15, fontweight="bold", y=1)
ax[0,0].text(max(xticks0) - 0.05 * max(xticks0), max(yticks0) - 0.01 * max(yticks0), text00, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
ax[0,0].plot([min(xticks0), max(xticks0)], [min(yticks0), max(yticks0)], 'k--')
#################################
ax[0,1].set_box_aspect(1)
sns.set_context("poster")

sns.regplot(x=male_post["actual_hgs"], y=male_post["predicted_hgs"], ax=ax[0,1], line_kws={"color": "red"})

ax[0,1].tick_params(axis='both', labelsize=20)

xmin0, xmax0 = ax[0,1].get_xlim()
ymin0, ymax0 = ax[0,1].get_ylim()
max0 = max(xmax0, ymax0)
min0 = min(xmin0, ymin0)
ax[0,1].set_ylim(min0, max0)
ax[0,1].set_xlim(min0, max0)
xmin0, xmax0 = ax[0,1].get_xlim()
ymin0, ymax0 = ax[0,1].get_ylim()
yticks0 = ax[0,1].get_yticks()
xticks0 = ax[0,1].get_xticks()
ax[0,1].set_xticks(yticks0)
ax[0,1].set_yticks(xticks0)
text01 = 'CORR: ' + str(format(corr_male_post, '.3f'))
ax[0,1].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
ax[0,1].set_ylabel('Predicted HGS', fontsize=20, fontweight="bold")

ax[0,1].set_title(f"Predicted vs Actual HGS - Post-stroke - Males(N={len(male_post)})", fontsize=15, fontweight="bold", y=1)
ax[0,1].text(max(xticks0) - 0.05 * max(xticks0), max(yticks0) - 0.01 * max(yticks0), text01, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
ax[0,1].plot([min(xticks0), max(xticks0)], [min(yticks0), max(yticks0)], 'k--')
#################################
ax[1,0].set_box_aspect(1)
sns.set_context("poster")
sns.regplot(x=female_pre["actual_hgs"], y=female_pre["predicted_hgs"], ax=ax[1,0], scatter_kws={"color": "grey"}, line_kws={"color": "red"})
ax[1,0].tick_params(axis='both', labelsize=20)

xmin1, xmax1 = ax[1,0].get_xlim()
ymin1, ymax1 = ax[1,0].get_ylim()
max1 = max(xmax1, ymax1)
min1 = min(xmin1, ymin1)
ax[1,0].set_xlim(min1, max1)
ax[1,0].set_ylim(min1, max1)
xmin1, xmax1 = ax[1,0].get_xlim()
ymin1, ymax1 = ax[1,0].get_ylim()

yticks1 = ax[1,0].get_yticks()
xticks1 = ax[1,0].get_xticks()
ax[1,0].set_xticks(yticks1, fontsize=20)
ax[1,0].set_yticks(xticks1, fontsize=20)

text10 = 'CORR: ' + str(format(corr_female_pre, '.3f'))
# ax2.xlabel('Correlation of LCOR with Actual HGS', fontsize=25, fontweight="bold")
# ax2.ylabel('Correlation of LCOR with Predicted HGS', fontsize=25, fontweight="bold")
ax[1,0].set_title(f"Predicted vs Actual HGS - Pre-stroke - Females(N={len(female_post)})", fontsize=15, fontweight="bold", y=1)
ax[1,0].text(max(xticks1) - 0.05 * max(xticks1), max(yticks1) - 0.01 * max(yticks1), text10, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# Add a diagonal line
ax[1,0].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
ax[1,0].set_ylabel('Predicted HGS', fontsize=20, fontweight="bold")
ax[1,0].plot([min(xticks1), max(xticks1)], [min(yticks1), max(yticks1)], 'k--')
#################################
ax[1,1].set_box_aspect(1)
sns.set_context("poster")
sns.regplot(x=female_post["actual_hgs"], y=female_post["predicted_hgs"], ax=ax[1,1], scatter_kws={"color": "orange"}, line_kws={"color": "red"})
ax[1,1].tick_params(axis='both', labelsize=20)

xmin1, xmax1 = ax[1,1].get_xlim()
ymin1, ymax1 = ax[1,1].get_ylim()
max1 = max(xmax1, ymax1)
min1 = min(xmin1, ymin1)
ax[1,1].set_xlim(min1, max1)
ax[1,1].set_ylim(min1, max1)
xmin1, xmax1 = ax[1,1].get_xlim()
ymin1, ymax1 = ax[1,1].get_ylim()

yticks1 = ax[1,1].get_yticks()
xticks1 = ax[1,1].get_xticks()
ax[1,1].set_xticks(yticks1, fontsize=20)
ax[1,1].set_yticks(xticks1, fontsize=20)

text11 = 'CORR: ' + str(format(corr_female_post, '.3f'))
# ax2.xlabel('Correlation of LCOR with Actual HGS', fontsize=25, fontweight="bold")
# ax2.ylabel('Correlation of LCOR with Predicted HGS', fontsize=25, fontweight="bold")
ax[1,1].set_title(f"Predicted vs Actual HGS - Post-stroke - Females(N={len(female_post)})", fontsize=15, fontweight="bold", y=1)
ax[1,1].text(max(xticks1) - 0.05 * max(xticks1), max(yticks1) - 0.01 * max(yticks1), text11, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# Add a diagonal line
ax[1,1].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
ax[1,1].set_ylabel('Predicted HGS', fontsize=20, fontweight="bold")
ax[1,1].plot([min(xticks1), max(xticks1)], [min(yticks1), max(yticks1)], 'k--')

plt.suptitle("Predicted vs Actual HGS - Longitudinal Stroke MRI(N=76)", fontsize=20, fontweight="bold", y=0.95)

plt.show()
plt.savefig(f"correlate_actual_predicted_hgs_storke_pre_post.png")
plt.close()
##############################################################################################################################################################
# Pre-Post predicted HGS vs True HGS:
female_diff_predicted = female_pre['predicted_hgs'] - female_post['predicted_hgs']
male_diff_predicted = male_pre['predicted_hgs'] - male_post['predicted_hgs']

corr_m_pre, p_m_pre = spearmanr(male_diff_predicted, male_pre['actual_hgs'])
corr_m_post, p_m_post = spearmanr(male_diff_predicted, male_post['actual_hgs'])

corr_f_pre, p_f_pre = spearmanr(female_diff_predicted, female_pre['actual_hgs'])
corr_f_post, p_f_post = spearmanr(female_diff_predicted, female_post['actual_hgs'])

##############################################################################################################################################################
corr_m_pre, p_m_pre = spearmanr(male_pre['predicted_hgs'], male_pre['actual_hgs'])
corr_m_post, p_m_post = spearmanr(male_post['predicted_hgs'], male_pre['actual_hgs'])

corr_f_pre, p_f_pre = spearmanr(female_pre['predicted_hgs'], female_pre['actual_hgs'])
corr_f_post, p_f_post = spearmanr(female_post['predicted_hgs'], female_pre['actual_hgs'])
##############################################################################################################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
fig, ax = plt.subplots(2, 2, figsize=(25,25))
sns.set_context("poster")
# plt.plot(y_true_fefemale, y_pred_fefemale)
# plt.axis("equal")
ax[0,0].set_box_aspect(1)

sns.regplot(x=male_pre["actual_hgs"], y=male_pre["predicted_hgs"], ax=ax[0,0], line_kws={"color": "red"}, scatter_kws={"color": "grey"})

ax[0,0].tick_params(axis='both', labelsize=20)

xmin0, xmax0 = ax[0,0].get_xlim()
ymin0, ymax0 = ax[0,0].get_ylim()
max0 = max(xmax0, ymax0)
min0 = min(xmin0, ymin0)
ax[0,0].set_ylim(min0, max0)
ax[0,0].set_xlim(min0, max0)
xmin0, xmax0 = ax[0,0].get_xlim()
ymin0, ymax0 = ax[0,0].get_ylim()
yticks0 = ax[0,0].get_yticks()
xticks0 = ax[0,0].get_xticks()
ax[0,0].set_xticks(yticks0)
ax[0,0].set_yticks(xticks0)
text00 = 'CORR: ' + str(format(corr_m_pre, '.3f'))
# ax[0,0].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
# ax[0,0].set_ylabel('Predicted HGS difference(pre-stroke pred. - post-stroke pred.)', fontsize=20, fontweight="bold")
ax[0, 0].set(xlabel =None, ylabel=None)

ax[0,0].set_title(f"Predicted vs Actual HGS - Pre-stroke - Males(N={len(male_post)})", fontsize=20, fontweight="bold", y=1)
ax[0,0].text(max(xticks0) - 0.05 * max(xticks0), max(yticks0) - 0.01 * max(yticks0), text00, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
ax[0,0].plot([min(xticks0), max(xticks0)], [min(yticks0), max(yticks0)], 'k--')
#################################
ax[0,1].set_box_aspect(1)
sns.set_context("poster")

sns.regplot(x=male_pre["actual_hgs"], y=male_post["predicted_hgs"], ax=ax[0,1], line_kws={"color": "red"})

ax[0,1].tick_params(axis='both', labelsize=20)

xmin0, xmax0 = ax[0,1].get_xlim()
ymin0, ymax0 = ax[0,1].get_ylim()
max0 = max(xmax0, ymax0)
min0 = min(xmin0, ymin0)
ax[0,1].set_ylim(min0, max0)
ax[0,1].set_xlim(min0, max0)
xmin0, xmax0 = ax[0,1].get_xlim()
ymin0, ymax0 = ax[0,1].get_ylim()
yticks0 = ax[0,1].get_yticks()
xticks0 = ax[0,1].get_xticks()
ax[0,1].set_xticks(yticks0)
ax[0,1].set_yticks(xticks0)
text01 = 'CORR: ' + str(format(corr_m_post, '.3f'))
# ax[0,1].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
# ax[0,1].set_ylabel('Predicted HGS difference(pre-stroke pred. - post-stroke pred.)', fontsize=20, fontweight="bold")
ax[0,1].set(xlabel =None, ylabel=None)
ax[0,1].set_title(f"Predicted vs Actual HGS - Pre-stroke - Males(N={len(male_post)})", fontsize=20, fontweight="bold", y=1)
ax[0,1].text(max(xticks0) - 0.05 * max(xticks0), max(yticks0) - 0.01 * max(yticks0), text01, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
ax[0,1].plot([min(xticks0), max(xticks0)], [min(yticks0), max(yticks0)], 'k--')
#################################
ax[1,0].set_box_aspect(1)
sns.set_context("poster")
sns.regplot(x=female_pre["actual_hgs"], y=female_pre["predicted_hgs"], ax=ax[1,0], scatter_kws={"color": "grey"}, line_kws={"color": "red"})
ax[1,0].tick_params(axis='both', labelsize=20)

xmin1, xmax1 = ax[1,0].get_xlim()
ymin1, ymax1 = ax[1,0].get_ylim()
max1 = max(xmax1, ymax1)
min1 = min(xmin1, ymin1)
ax[1,0].set_xlim(min1, max1)
ax[1,0].set_ylim(min1, max1)
xmin1, xmax1 = ax[1,0].get_xlim()
ymin1, ymax1 = ax[1,0].get_ylim()

yticks1 = ax[1,0].get_yticks()
xticks1 = ax[1,0].get_xticks()
ax[1,0].set_xticks(yticks1, fontsize=20)
ax[1,0].set_yticks(xticks1, fontsize=20)

text10 = 'CORR: ' + str(format(corr_f_pre, '.3f'))
# ax2.xlabel('Correlation of LCOR with Actual HGS', fontsize=25, fontweight="bold")
# ax2.ylabel('Correlation of LCOR with Predicted HGS', fontsize=25, fontweight="bold")
ax[1,0].set_title(f"Predicted vs Actual HGS - Pre-stroke - Females(N={len(female_post)})", fontsize=20, fontweight="bold", y=1)
ax[1,0].text(max(xticks1) - 0.05 * max(xticks1), max(yticks1) - 0.01 * max(yticks1), text10, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# Add a diagonal line
# ax[1,0].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
# ax[1,0].set_ylabel('Predicted HGS difference(pre-stroke pred. - post-stroke pred.)', fontsize=20, fontweight="bold")
ax[1,0].set(xlabel =None, ylabel=None)
ax[1,0].plot([min(xticks1), max(xticks1)], [min(yticks1), max(yticks1)], 'k--')
#################################
ax[1,1].set_box_aspect(1)
sns.set_context("poster")
sns.regplot(x=female_pre["actual_hgs"], y=female_post["predicted_hgs"], ax=ax[1,1], scatter_kws={"color": "orange"}, line_kws={"color": "red"})
ax[1,1].tick_params(axis='both', labelsize=20)

xmin1, xmax1 = ax[1,1].get_xlim()
ymin1, ymax1 = ax[1,1].get_ylim()
max1 = max(xmax1, ymax1)
min1 = min(xmin1, ymin1)
ax[1,1].set_xlim(min1, max1)
ax[1,1].set_ylim(min1, max1)
xmin1, xmax1 = ax[1,1].get_xlim()
ymin1, ymax1 = ax[1,1].get_ylim()

yticks1 = ax[1,1].get_yticks()
xticks1 = ax[1,1].get_xticks()
ax[1,1].set_xticks(yticks1, fontsize=20)
ax[1,1].set_yticks(xticks1, fontsize=20)

text11 = 'CORR: ' + str(format(corr_f_post, '.3f'))
# ax2.xlabel('Correlation of LCOR with Actual HGS', fontsize=25, fontweight="bold")
# ax2.ylabel('Correlation of LCOR with Predicted HGS', fontsize=25, fontweight="bold")
ax[1,1].set_title(f"Predicted vs Actual HGS - Post-stroke - Females(N={len(female_post)})", fontsize=20, fontweight="bold", y=1)
ax[1,1].text(max(xticks1) - 0.05 * max(xticks1), max(yticks1) - 0.01 * max(yticks1), text11, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# Add a diagonal line
# ax[1,1].set_xlabel('Actual HGS', fontsize=20, fontweight="bold")
# ax[1,1].set_ylabel('Predicted HGS difference(pre-stroke pred. - post-stroke pred.)', fontsize=20, fontweight="bold")
ax[1, 1].set(xlabel =None, ylabel=None)
ax[1,1].plot([min(xticks1), max(xticks1)], [min(yticks1), max(yticks1)], 'k--')

# Set labels for x-axis and y-axis
fig.text(0.5, 0.07, f'Actual HGS (Pre-stroke)', ha='center', fontsize=30, fontweight="bold")
fig.text(0.07, 0.5, f'Predicted HGS', va='center',
        rotation='vertical', fontsize=30, fontweight="bold")
plt.suptitle("Predicted vs Actual HGS - Longitudinal Stroke MRI(N=76)", fontsize=25, fontweight="bold", y=0.95)

plt.show()
plt.savefig(f"correlate_actual_predicted_hgs_storke_pre_post_Predicted difference.png")
plt.close()


print("===== Done! =====")
embed(globals(), locals())
