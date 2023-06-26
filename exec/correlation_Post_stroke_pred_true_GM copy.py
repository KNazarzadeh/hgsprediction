
import os
import pandas as pd
from hgsprediction.input_arguments import parse_args, input_arguments
import numpy as np
import sys
import datatable as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import pickle
from LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc


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

##############################################################################
# GMV
jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "brain_data",
)

jay_file_1 = os.path.join(
        jay_path,
        '1_gmd_schaefer_all_subjects.jay')
jay_file_4 = os.path.join(
        jay_path,
        '4_gmd_tian_all_subjects.jay')
jay_file_2 = os.path.join(
        jay_path,
        '2_gmd_SUIT_all_subjects.jay')

# fname = base_dir / '1_gmd_schaefer_all_subjects.jay'
# feature_dt = dt.fread(jay_file.as_posix())
feature_dt_1 = dt.fread(jay_file_1)
feature_df_1 = feature_dt_1.to_pandas()
feature_df_1.set_index('SubjectID', inplace=True)

feature_dt_4 = dt.fread(jay_file_4)
feature_df_4 = feature_dt_4.to_pandas()
feature_df_4.set_index('SubjectID', inplace=True)

feature_dt_2 = dt.fread(jay_file_2)
feature_df_2 = feature_dt_2.to_pandas()
feature_df_2.set_index('SubjectID', inplace=True)

feature_df_1.index = feature_df_1.index.str.replace("sub-", "")
feature_df_1.index = feature_df_1.index.map(int)

feature_df_2.index = feature_df_2.index.str.replace("sub-", "")
feature_df_2.index = feature_df_2.index.map(int)

feature_df_4.index = feature_df_4.index.str.replace("sub-", "")
feature_df_4.index = feature_df_4.index.map(int)

gm_anthro_all = pd.concat([feature_df_1, feature_df_2, feature_df_4], axis=1)
gm_anthro_all = gm_anthro_all.dropna()

###############################################################################
post_list = ["1_post_session", "2_post_session", "3_post_session", "4_post_session"]

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

file_path = os.path.join(
        folder_path,
        f"{post_list[0]}_{mri_status}_{population}.csv")

df_post = pd.read_csv(file_path, sep=',')
df_post.set_index("SubjectID", inplace=True)

ses = df_post["1_post_session"].astype(str).str[8:]
for i in range(0, len(df_post["1_post_session"])):
    idx=ses.index[i]
    if ses.iloc[i] != "":
        df_post.loc[idx, "1_post_days"] = df_post.loc[idx, f"followup_days-{ses.iloc[i]}"]
    else:
        df_post.loc[idx, "1_post_days"] = np.NaN

##############################################################################
# Replace Age
for ses in range(0,4):
    sub_id = df_post[df_post['1_post_session']== f"session-{ses}.0"].index.values
    df_post.loc[sub_id, '1_post_age'] = df_post.loc[sub_id, f'21003-{ses}.0']
##############################################################################
# Replace BMI
for ses in range(0,4):
    sub_id = df_post[df_post['1_post_session']== f"session-{ses}.0"].index.values
    df_post.loc[sub_id, '1_post_bmi'] = df_post.loc[sub_id, f'21001-{ses}.0']
##############################################################################
# Replace Height
for ses in range(0,4):
    sub_id = df_post[df_post['1_post_session']== f"session-{ses}.0"].index.values
    df_post.loc[sub_id, '1_post_height'] = df_post.loc[sub_id, f'50-{ses}.0']
##############################################################################
# Replace waist to hip ratio
for ses in range(0,4):
    sub_id = df_post[df_post['1_post_session']== f"session-{ses}.0"].index.values
    df_post.loc[sub_id, '1_post_waist'] = df_post.loc[sub_id, f'48-{ses}.0']
    df_post.loc[sub_id, '1_post_hip'] = df_post.loc[sub_id, f'49-{ses}.0']

    df_post['1_post_waist_hip_ratio'] = (df_post.loc[:, "1_post_waist"].astype(str).astype(float)).div(
                df_post.loc[:, "1_post_hip"].astype(str).astype(float))
##############################################################################
sub_id = df_post[df_post['1707-0.0']== 1.0].index.values
# Add and new column "dominant_hgs"
# And assign Right hand HGS value:
df_post.loc[sub_id, "dominant_hgs"] = \
    df_post.loc[sub_id, "1_post_right_hgs"]
df_post.loc[sub_id, "nondominant_hgs"] = \
    df_post.loc[sub_id, "1_post_left_hgs"]
# ------------------------------------
# If handedness is equal to 2
# Left hand is Dominantsession
# Find handedness equal to 2:
sub_id = df_post[df_post['1707-0.0']== 2.0].index.values
# Add and new column "dominant_hgs"
# And assign Left hand HGS value:
df_post.loc[sub_id, "dominant_hgs"] = \
    df_post.loc[sub_id, "1_post_left_hgs"]
df_post.loc[sub_id, "nondominant_hgs"] = \
    df_post.loc[sub_id, "1_post_right_hgs"]
# ------------------------------------
# If handedness is equal to:
# 3 (Use both right and left hands equally) OR
# -3 (handiness is not available/Prefer not to answer) OR
# NaN value
# Dominant will be the Highest Handgrip score from both hands.
# Find handedness equal to 3, -3 or NaN:
sub_id = df_post[(df_post['1707-0.0']== 3.0) | (df_post['1707-0.0']== -3.0) | (df_post['1707-0.0'].isna())].index.values
# Add and new column "dominant_hgs"
# And assign Highest HGS value among Right and Left HGS:        
df_post.loc[sub_id, f"dominant_hgs"] = df_post.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].max(axis=1)
df_post.loc[sub_id, f"nondominant_hgs"] = df_post.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].min(axis=1)

###############################################################################
df_post.loc[:, f"1_post_hgs(L+R)"] = \
            df_post.loc[:, f"1_post_left_hgs"] + df_post.loc[:, f"1_post_right_hgs"]
            
df_post = df_post[df_post.loc[:, f"dominant_hgs"] >=4]

##############################################################################
# extract_features = ExtractFeatures(df_tmp_mri, motor, population)
# extracted_data = extract_features.extract_features()
# Remove columns that all values are NaN
nan_cols = df_post.columns[df_post.isna().all()].tolist()
df_test_set = df_post.drop(nan_cols, axis=1)

mri_features = df_test_set[(df_test_set['1_post_session']=="session-2.0") | (df_test_set['1_post_session']=="session-3.0")]
    
# X = define_features(feature_type, new_data)
X = ["1_post_age", "1_post_bmi", "1_post_height", "1_post_waist_hip_ratio"]
# Target: HGS(L+R)
# y = define_target(target)
if target == "L+R":
    y = "1_post_hgs(L+R)"

###############################################################################
# Remove Missing data from Features and Target
mri_features = mri_features.dropna(subset=y)
mri_features = mri_features.dropna(subset=X)

new_data = mri_features[X]
new_data = new_data.rename(columns={'1_post_age': 'Age1stVisit', '1_post_bmi': '21001-0.0', '1_post_height':'50-0.0', '1_post_waist_hip_ratio': 'waist_to_hip_ratio-0.0'})
new_data = pd.concat([new_data, mri_features[y],mri_features['31-0.0']], axis=1)

df_test_female = new_data[new_data['31-0.0']==0]
df_test_male = new_data[new_data['31-0.0']==1]

X = ['Age1stVisit', '21001-0.0', '50-0.0', 'waist_to_hip_ratio-0.0']

###############################################################################
gm_anthro_age_stroke = gm_anthro_all[gm_anthro_all.index.isin(new_data.index)]
stroke_no_gm = new_data[~new_data.index.isin(gm_anthro_age_stroke.index)]
gm_anthro_female = gm_anthro_age_stroke[gm_anthro_age_stroke.index.isin(df_test_female.index)]
gm_anthro_male = gm_anthro_age_stroke[gm_anthro_age_stroke.index.isin(df_test_male.index)]

mri_features = mri_features[mri_features.index.isin(gm_anthro_age_stroke.index)]
df_test_female = df_test_female[df_test_female.index.isin(gm_anthro_female.index)]
df_test_male = df_test_male[df_test_male.index.isin(gm_anthro_male.index)]

f_days = mri_features[mri_features['31-0.0']==0.0]['1_post_days']
f_hgs_LR = df_test_female["1_post_hgs(L+R)"]

m_days = mri_features[mri_features['31-0.0']==1.0]['1_post_days']
m_hgs_LR = df_test_male["1_post_hgs(L+R)"]

###############################################################################
# Female
y_true_female = df_test_female[y]
y_pred_female = model_trained_female.predict(df_test_female[X])
df_test_female["actual_hgs"] = y_true_female
df_test_female["predicted_hgs"] = y_pred_female
df_test_female["hgs_diff"] = abs(y_true_female - y_pred_female)
corr_female_diff, p_female_diff = spearmanr(abs(df_test_female["hgs_diff"]), f_days)
# corr_female_diff = format(np.corrcoef(df_test_female["hgs_diff"], f_days)[1, 0], '.2f')

###############################################################################
# Male
y_true_male = df_test_male[y]
y_pred_male = model_trained_male.predict(df_test_male[X])
df_test_male["actual_hgs"] = y_true_male
df_test_male["predicted_hgs"] = y_pred_male
df_test_male["hgs_diff"] = abs(y_true_male - y_pred_male)
corr_male_diff, p_male_diff = spearmanr(abs(df_test_male["hgs_diff"]), m_days)
# corr_male_diff = format(np.corrcoef(df_test_male["hgs_diff"], m_days)[1, 0], '.2f')
###############################################################################
# Females
# Calculate correlations between gm and Predicted HGS
gm_region_corr_predicted_female = pd.DataFrame(columns=gm_anthro_female.columns)

n_regions = 1088
correlations_female = np.zeros(n_regions)
p_values_female = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_female.iloc[:, region], df_test_female['predicted_hgs'])
    correlations_female[region] = corr
    p_values_female[region] = p_value

gm_region_corr_predicted_female.loc['correlations',:]= correlations_female.tolist()
gm_region_corr_predicted_female.loc['p_values',:]= p_values_female.tolist()

# Print correlations and p-values for each region
for region in range(n_regions):
    print(f"{gm_anthro_female.columns[region]}: Correlation_female = {correlations_female[region]:.3f}, p-value_female = {p_values_female[region]:.3f}")
###############################################################################
# Males
# Calculate correlations between gm and Predicted HGS
gm_region_corr_predicted_male = pd.DataFrame(columns=gm_anthro_male.columns)

n_regions = 1088
correlations_male = np.zeros(n_regions)
p_values_male = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_male.iloc[:, region], df_test_male['predicted_hgs'])
    correlations_male[region] = corr
    p_values_male[region] = p_value
gm_region_corr_predicted_male.loc['correlations',:]= correlations_male.tolist()
gm_region_corr_predicted_male.loc['p_values',:]= p_values_male.tolist()

# Print correlations and p-values for each region
for region in range(n_regions):
    print(f"{gm_anthro_male.columns[region]}: Correlation_male = {correlations_male[region]:.3f}, p-value_male = {p_values_male[region]:.3f}")
    

###############################################################################
df_predicted_male = gm_region_corr_predicted_male.T
# cast the strings to floats
df_predicted_male['correlations'] = abs(df_predicted_male['correlations'].astype(float))
top10_predicted_male = df_predicted_male.nlargest(10, 'correlations')
highest_region_predicted_male = top10_predicted_male.iloc[0].name

df_predicted_female = gm_region_corr_predicted_female.T
# cast the strings to floats
df_predicted_female['correlations'] = abs(df_predicted_female['correlations'].astype(float))
top10_predicted_female = df_predicted_female.nlargest(10, 'correlations')
highest_region_predicted_female = top10_predicted_female.iloc[0].name
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# Females
# Calculate correlations between gm and Predicted - Actual HGS
gm_region_corr_actual_predicted_female = pd.DataFrame(columns=gm_anthro_female.columns)

n_regions = 1088
correlations_female = np.zeros(n_regions)
p_values_female = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_female.iloc[:, region], df_test_female['hgs_diff'])
    correlations_female[region] = corr
    p_values_female[region] = p_value

gm_region_corr_actual_predicted_female.loc['correlations',:]= correlations_female.tolist()
gm_region_corr_actual_predicted_female.loc['p_values',:]= p_values_female.tolist()

# Print correlations and p-values for each region
for region in range(n_regions):
    print(f"{gm_anthro_female.columns[region]}: Correlation_female = {correlations_female[region]:.3f}, p-value_female = {p_values_female[region]:.3f}")
###############################################################################
# Males
# Calculate correlations between gm and Predicted - Actual HGS
gm_region_corr_actual_predicted_male = pd.DataFrame(columns=gm_anthro_male.columns)

n_regions = 1088
correlations_male = np.zeros(n_regions)
p_values_male = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_male.iloc[:, region], df_test_male['hgs_diff'])
    correlations_male[region] = corr
    p_values_male[region] = p_value
gm_region_corr_actual_predicted_male.loc['correlations',:]= correlations_male.tolist()
gm_region_corr_actual_predicted_male.loc['p_values',:]= p_values_male.tolist()

# Print correlations and p-values for each region
for region in range(n_regions):
    print(f"{gm_anthro_male.columns[region]}: Correlation_male = {correlations_male[region]:.3f}, p-value_male = {p_values_male[region]:.3f}")
    
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
df_actual_predicted_male = gm_region_corr_actual_predicted_male.T
# cast the strings to floats
df_actual_predicted_male['correlations'] = abs(df_actual_predicted_male['correlations'].astype(float))
top10_actual_predicted_male = df_actual_predicted_male.nlargest(10, 'correlations')
highest_region_actual_predicted_male = top10_actual_predicted_male.iloc[0].name

df_actual_predicted_female = gm_region_corr_actual_predicted_female.T
# cast the strings to floats
df_actual_predicted_female['correlations'] = abs(df_actual_predicted_female['correlations'].astype(float))
top10_actual_predicted_female = df_actual_predicted_female.nlargest(10, 'correlations')
highest_region_actual_predicted_female = top10_actual_predicted_female.iloc[0].name
###############################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
fig, ax = plt.subplots(1, 2, figsize=(20,10))
sns.set_context("poster")
ax[0].set_box_aspect(1)
sns.regplot(x=m_days, y=gm_region_corr_predicted_male.T['correlations'], ax=ax[0], line_kws={"color": "red"})
ax[0].tick_params(axis='both', labelsize=20)

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

# xmax0 = xmax0+10
# ymax0 = ymax0+10

ax[0].set_xlim(0, xmax0)
ax[0].set_ylim(0, ymax0)

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

text0 = 'CORR: ' + str(format(corr_male_diff, '.3f'))
ax[0].set_xlabel('Post-stroke Days', fontsize=20, fontweight="bold")
ax[0].set_ylabel('Difference (Actual-Predicted)HGS', fontsize=20, fontweight="bold")

ax[0].set_title(f"Males({len(df_test_male)})", fontsize=15, fontweight="bold", y=1)
# ax[0].text(xmax0 - 0.05 * xmax0, ymax0 - 0.01 * ymax0, text0, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
# ax[0].plot([xmin0, xmax0], [ymin0, ymax0], 'k--')

#################################
ax[1].set_box_aspect(1)
# sns.set_context("poster")
sns.regplot(x=f_days, y=gm_region_corr_predicted_female.T['correlations'], ax=ax[1], scatter_kws={"color": "orange"}, line_kws={"color": "red"})
ax[1].tick_params(axis='both', labelsize=20)

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()
# xmax1 = xmax1+10
# ymax1 = ymax1+10

ax[1].set_xlim(0, xmax1)
ax[1].set_ylim(0, ymax1)

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()


text1 = 'CORR: ' + str(format(corr_female_diff, '.3f'))
ax[1].set_title(f"Females({len(df_test_female)})", fontsize=15, fontweight="bold", y=1)
# ax[1].text(xmax1 - 0.05 * xmax1, ymax1 - 0.01 * ymax1, text1, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")

# Add a diagonal line
ax[1].set_xlabel('Post-stroke Days', fontsize=20, fontweight="bold")
ax[1].set_ylabel('Difference (Actual-Predicted)HGS', fontsize=20, fontweight="bold")

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()
xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()

xmin = min(xmin0, xmin1)
xmax = max(xmax0, xmax1)
ymin = min(ymin0, ymin1)
ymax = max(ymax0, ymax1)


ax[0].set_xlim(0, xmax)
ax[0].set_ylim(0, ymax)

ax[1].set_xlim(0, xmax)
ax[1].set_ylim(0, ymax)

ax[0].plot([xmin, xmax], [ymin, ymax], 'k--')
ax[1].plot([xmin, xmax], [ymin, ymax], 'k--')

ax[0].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text0, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")
ax[1].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text1, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

plt.suptitle("Difference (Actual-Predicted)HGS vs Post-stroke Days", fontsize=20, fontweight="bold", y=0.95)

plt.show()
plt.savefig(f"correlate_actual_predicted_hgs_storke_mri_both_gender_post_diff.png")
plt.close()

# ##############################################################################################################################################################
###############################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
fig, ax = plt.subplots(1, 2, figsize=(20,10))
sns.set_context("poster")
ax[0].set_box_aspect(1)
sns.regplot(x=m_days/365, y=df_test_male["hgs_diff"], ax=ax[0], line_kws={"color": "red"})
ax[0].tick_params(axis='both', labelsize=20)

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

# # xmax0 = xmax0+2
# ymax0 = ymax0+5

ax[0].set_xlim(0, xmax0)
ax[0].set_ylim(0, ymax0)

# xmin0, xmax0 = ax[0].get_xlim()
# ymin0, ymax0 = ax[0].get_ylim()

text0 = 'CORR: ' + str(format(corr_male_diff, '.3f'))
ax[0].set_xlabel('Post-stroke years', fontsize=20, fontweight="bold")
ax[0].set_ylabel('Difference (Actual-Predicted)HGS', fontsize=20, fontweight="bold")

ax[0].set_title(f"Males({len(df_test_male)})", fontsize=15, fontweight="bold", y=1)
# ax[0].text(xmax0 - 0.05 * xmax0, ymax0 - 0.01 * ymax0, text0, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
# ax[0].plot([xmin0, xmax0], [ymin0, ymax0], 'k--')

#################################
ax[1].set_box_aspect(1)
# sns.set_context("poster")
sns.regplot(x=f_days/365, y=df_test_female["hgs_diff"], ax=ax[1], scatter_kws={"color": "orange"}, line_kws={"color": "red"})
ax[1].tick_params(axis='both', labelsize=20)

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()
# xmax1 = xmax1+2
# ymax1 = ymax1+5

ax[1].set_xlim(0, xmax1)
ax[1].set_ylim(0, ymax1)

# xmin1, xmax1 = ax[1].get_xlim()
# ymin1, ymax1 = ax[1].get_ylim()


text1 = 'CORR: ' + str(format(corr_female_diff, '.3f'))
ax[1].set_title(f"Females({len(df_test_female)})", fontsize=15, fontweight="bold", y=1)


# Add a diagonal line
ax[1].set_xlabel('Post-stroke years', fontsize=20, fontweight="bold")
ax[1].set_ylabel('Difference (Actual-Predicted)HGS', fontsize=20, fontweight="bold")


xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()
xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()

xmin = min(xmin0, xmin1)
xmax = max(xmax0, xmax1)
ymin = min(ymin0, ymin1)
ymax = max(ymax0, ymax1)


ax[0].set_xlim(0, xmax)
ax[0].set_ylim(0, ymax)

ax[1].set_xlim(0, xmax)
ax[1].set_ylim(0, ymax)

ax[0].plot([xmin, xmax], [ymin, ymax], 'k--')
ax[1].plot([xmin, xmax], [ymin, ymax], 'k--')

ax[0].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text0, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")
ax[1].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text1, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

plt.suptitle("Difference (Actual-Predicted)HGS vs Post-stroke years", fontsize=20, fontweight="bold", y=0.95)

plt.show()
plt.savefig(f"correlate_actual_predicted_hgs_storke_mri_both_gender_post_diff_years.png")
plt.close()
