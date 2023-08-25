#!/home/knazarzadeh/miniconda3/envs/disorderspredwp3/bin/python3
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
from pingouin import partial_corr

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model_type, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)

###############################################################################

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

# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
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
df_ses3 = df_post[df_post["1_post_session"] == "session-3.0"]
df_ses2 = df_post[~df_post.index.isin(df_ses3.index)]
# print("===== Done! =====")
# embed(globals(), locals())
# Replace Age
df_ses3.loc[:, 'post_age'] = df_ses3.loc[:, f'21003-3.0']
##############################################################################
# Replace BMI
df_ses3.loc[:, 'post_bmi'] = df_ses3.loc[:, f'21001-3.0']
##############################################################################
# Replace Height
df_ses3.loc[:, 'post_height'] = df_ses3.loc[:, f'50-3.0']

df_ses3.loc[:, 'post_days'] = df_ses3.loc[:, f'followup_days-3.0']

##############################################################################
# Replace waist to hip ratio
df_ses3.loc[:, 'post_waist'] = df_ses3.loc[:, f'48-3.0']
df_ses3.loc[:, 'post_hip'] = df_ses3.loc[:, f'49-3.0']

df_ses3['post_waist_hip_ratio'] = (df_ses3.loc[:, "post_waist"].astype(str).astype(float)).div(
                df_ses3.loc[:, "post_hip"].astype(str).astype(float))
##############################################################################
# Replace Age
df_ses2.loc[:, 'post_age'] = df_ses2.loc[:, f'21003-2.0']
##############################################################################
# Replace BMI
df_ses2.loc[:, 'post_bmi'] = df_ses2.loc[:, f'21001-2.0']
##############################################################################
# Replace Height
df_ses2.loc[:, 'post_height'] = df_ses2.loc[:, f'50-2.0']

df_ses2.loc[:, 'post_days'] = df_ses2.loc[:, f'followup_days-2.0']

##############################################################################
# Replace waist to hip ratio
df_ses2.loc[:, 'post_waist'] = df_ses2.loc[:, f'48-2.0']
df_ses2.loc[:, 'post_hip'] = df_ses2.loc[:, f'49-2.0']

df_ses2['post_waist_hip_ratio'] = (df_ses2.loc[:, "post_waist"].astype(str).astype(float)).div(
                df_ses2.loc[:, "post_hip"].astype(str).astype(float))
##############################################################################
sub_id = df_ses2[df_ses2['1707-0.0']== 1.0].index.values
# Add and new column "dominant_hgs"
# And assign Right hand HGS value:
df_ses2.loc[sub_id, "dominant_hgs"] = \
    df_ses2.loc[sub_id, "1_post_right_hgs"]
df_ses2.loc[sub_id, "nondominant_hgs"] = \
    df_ses2.loc[sub_id, "1_post_left_hgs"]
# ------------------------------------
# If handedness is equal to 2
# Left hand is Dominantsession
# Find handedness equal to 2:
sub_id = df_ses2[df_ses2['1707-0.0']== 2.0].index.values
# Add and new column "dominant_hgs"
# And assign Left hand HGS value:
df_ses2.loc[sub_id, "dominant_hgs"] = \
    df_ses2.loc[sub_id, "1_post_left_hgs"]
df_ses2.loc[sub_id, "nondominant_hgs"] = \
    df_ses2.loc[sub_id, "1_post_right_hgs"]
# ------------------------------------
# If handedness is equal to:
# 3 (Use both right and left hands equally) OR
# -3 (handiness is not available/Prefer not to answer) OR
# NaN value
# Dominant will be the Highest Handgrip score from both hands.
# Find handedness equal to 3, -3 or NaN:
sub_id = df_ses2[(df_ses2['1707-0.0']== 3.0) | (df_ses2['1707-0.0']== -3.0) | (df_ses2['1707-0.0'].isna())].index.values
# Add and new column "dominant_hgs"
# And assign Highest HGS value among Right and Left HGS:        
df_ses2.loc[sub_id, f"dominant_hgs"] = df_ses2.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].max(axis=1)
df_ses2.loc[sub_id, f"nondominant_hgs"] = df_ses2.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].min(axis=1)

###############################################################################
##############################################################################
sub_id = df_ses3[df_ses3['1707-0.0']== 1.0].index.values
# Add and new column "dominant_hgs"
# And assign Right hand HGS value:
df_ses3.loc[sub_id, "dominant_hgs"] = \
    df_ses3.loc[sub_id, "1_post_right_hgs"]
df_ses3.loc[sub_id, "nondominant_hgs"] = \
    df_ses3.loc[sub_id, "1_post_left_hgs"]
# ------------------------------------
# If handedness is equal to 2
# Left hand is Dominantsession
# Find handedness equal to 2:
sub_id = df_ses3[df_ses3['1707-0.0']== 2.0].index.values
# Add and new column "dominant_hgs"
# And assign Left hand HGS value:
df_ses3.loc[sub_id, "dominant_hgs"] = \
    df_ses3.loc[sub_id, "1_post_left_hgs"]
df_ses3.loc[sub_id, "nondominant_hgs"] = \
    df_ses3.loc[sub_id, "1_post_right_hgs"]
# ------------------------------------
# If handedness is equal to:
# 3 (Use both right and left hands equally) OR
# -3 (handiness is not available/Prefer not to answer) OR
# NaN value
# Dominant will be the Highest Handgrip score from both hands.
# Find handedness equal to 3, -3 or NaN:
sub_id = df_ses3[(df_ses3['1707-0.0']== 3.0) | (df_ses3['1707-0.0']== -3.0) | (df_ses3['1707-0.0'].isna())].index.values
# Add and new column "dominant_hgs"
# And assign Highest HGS value among Right and Left HGS:        
df_ses3.loc[sub_id, f"dominant_hgs"] = df_ses3.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].max(axis=1)
df_ses3.loc[sub_id, f"nondominant_hgs"] = df_ses3.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].min(axis=1)
##############################################################################
df_ses2.loc[:, f"post_hgs(L+R)"] = \
            df_ses2.loc[:, f"46-2.0"] + df_ses2.loc[:, f"47-2.0"]

df_ses3.loc[:, f"post_hgs(L+R)"] = \
            df_ses3.loc[:, f"46-3.0"] + df_ses3.loc[:, f"47-3.0"]

df_post = pd.concat([df_ses2, df_ses3], axis=0)
df_post = df_post[df_post.loc[:, f"dominant_hgs"] >=4]

##############################################################################
# extract_features = ExtractFeatures(df_tmp_mri, motor, population)
# extracted_data = extract_features.extract_features()
# Remove columns that all values are NaN
nan_cols = df_post.columns[df_post.isna().all()].tolist()
df_test_set = df_post.drop(nan_cols, axis=1)

mri_features = df_test_set.copy()
    
# X = define_features(feature_type, new_data)
X = ["post_age", "post_bmi", "post_height", "post_waist_hip_ratio"]
# Target: HGS(L+R)
# y = define_target(target)
if target == "L+R":
    y = "post_hgs(L+R)"

###############################################################################
# Remove Missing data from Features and Target
mri_features = mri_features.dropna(subset=y)
mri_features = mri_features.dropna(subset=X)

new_data = mri_features[X]
new_data = new_data.rename(columns={'post_age': 'Age1stVisit', 'post_bmi': '21001-0.0', 'post_height':'50-0.0', 'post_waist_hip_ratio': 'WHR-0.0'})
new_data = pd.concat([new_data, mri_features[y],mri_features['31-0.0']], axis=1)

df_test_female = new_data[new_data['31-0.0']==0]
df_test_male = new_data[new_data['31-0.0']==1]

X = ['Age1stVisit', '21001-0.0', '50-0.0', 'WHR-0.0']

###############################################################################
# Both gender
# y_true = new_data[y]
# y_pred = model_trained.predict(new_data[X])
# # mri_df = pd.DataFrame()
# # mri_df["SubjectID"] = new_data['eid']
# new_data["actual_hgs"] = y_true
# new_data["predicted_hgs"] = y_pred
# # new_data["hgs_diff"] = y_true - y_pred
# mae = format(mean_absolute_error(y_true, y_pred), '.2f')
# corr = format(np.corrcoef(y_pred, y_true)[1, 0], '.2f')
# score = format(r2_score(y_true, y_pred), '.2f')

##############################################################################
# Female
y_true_female = df_test_female[y]
y_pred_female = model_trained_female.predict(df_test_female[X])
df_test_female["actual_hgs"] = y_true_female
df_test_female["predicted_hgs"] = y_pred_female
# df_test_female["hgs_diff"] = y_true_female - y_pred_female
corr_female = format(np.corrcoef(y_pred_female, y_true_female)[1, 0], '.2f')
###############################################################################
# Male
y_true_male = df_test_male[y]
y_pred_male = model_trained_male.predict(df_test_male[X])
df_test_male["actual_hgs"] = y_true_male
df_test_male["predicted_hgs"] = y_pred_male
# df_test_male["hgs_diff"] = y_true_male - y_pred_male
corr_male = format(np.corrcoef(y_pred_male, y_true_male)[1, 0], '.2f')

##############################################################################
# y_diff_both = y_true - y_pred
# y_diff_female = y_true_female - y_pred_female
# y_diff_male = y_true_male - y_pred_male
# age_female = df_test_female['AgeAtScan']
# age_male = df_test_male['AgeAtScan']
# corr_female_age = format(np.corrcoef(y_diff_female, age_female)[1, 0], '.3f')
# corr_male_age = format(np.corrcoef(y_diff_male, age_male)[1, 0], '.3f')
###############################################################################
jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "brain_data",
)

jay_file_1 = os.path.join(
        jay_path,
        'GCOR_Schaefer400x7_Mean.jay')
# jay_file_2 = os.path.join(
#         jay_path,
#         'GCOR_Tian_Mean.jay')

# fname = base_dir / '1_gmd_schaefer_all_subjects.jay'
# feature_dt = dt.fread(jay_file.as_posix())
feature_dt_1 = dt.fread(jay_file_1)
feature_df_1 = feature_dt_1.to_pandas()
feature_df_1.set_index('SubjectID', inplace=True)

# feature_dt_2 = dt.fread(jay_file_2)
# feature_df_2 = feature_dt_2.to_pandas()
# feature_df_2.set_index('SubjectID', inplace=True)

feature_df_1.index = feature_df_1.index.str.replace("sub-", "")
feature_df_1.index = feature_df_1.index.map(int)

# feature_df_2.index = feature_df_2.index.str.replace("sub-", "")
# feature_df_2.index = feature_df_2.index.map(int)

# gm_anthro_all = pd.concat([feature_df_1, feature_df_2], axis=1)
gm_anthro_all = feature_df_1.copy()
gm_anthro_all = gm_anthro_all.dropna()
# print("===== Done! =====")
# embed(globals(), locals())
# Remove stroke subjects

##############################################################################
gm_anthro_no_stroke = gm_anthro_all[gm_anthro_all.index.isin(new_data.index)]
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
gm_anthro = gm_anthro_no_stroke[gm_anthro_no_stroke.index.isin(new_data.index)]
mri_anthro = new_data[new_data.index.isin(gm_anthro.index)]
# mri_anthro = mri_anthro.set_index('eid')
# mri_anthro = mri_anthro.reindex(gm_anthro.index)
# mri_anthro = mri_anthro.rename(index={'eid': 'SubjectID'})

gm_anthro_female = gm_anthro_no_stroke[gm_anthro_no_stroke.index.isin(df_test_female.index)]
mri_anthro_female = df_test_female[df_test_female.index.isin(gm_anthro.index)]
# mri_anthro_female = mri_anthro_female.set_index('eid')
# mri_anthro_female = mri_anthro_female.reindex(gm_anthro_female.index)
# mri_anthro_female = mri_anthro_female.rename(index={'eid': 'SubjectID'})

gm_anthro_male = gm_anthro_no_stroke[gm_anthro_no_stroke.index.isin(df_test_male.index)]
mri_anthro_male = df_test_male[df_test_male.index.isin(gm_anthro.index)]
# mri_anthro_male = mri_anthro_male.set_index('eid')
# mri_anthro_male = mri_anthro_male.reindex(gm_anthro_male.index)
# mri_anthro_male = mri_anthro_male.rename(index={'eid': 'SubjectID'})


##############################################################################
# Calculate correlations between gm and predicted HGS
# gm_region_corr_predicted = pd.DataFrame(columns=gm_anthro.columns)

# n_regions = 400
# n_subs = len(gm_anthro)
# correlations = np.zeros(n_regions)
# p_values = np.zeros(n_regions)
# for region in range(n_regions):
#     corr, p_value = spearmanr(gm_anthro.iloc[:, region], mri_anthro['predicted_hgs'])
#     correlations[region] = corr
#     p_values[region] = p_value

# gm_region_corr_predicted.loc['correlations',:]= correlations.tolist()
# gm_region_corr_predicted.loc['p_values',:]= p_values.tolist()

# # Print correlations and p-values for each region
# for region in range(n_regions):
#     print(f"{gm_anthro.columns[region]}: Correlation = {correlations[region]:.3f}, p-value = {p_values[region]:.3f}")
###############################################################################
# Females
# Calculate correlations between gm and Predicted HGS
gm_region_corr_predicted_female = pd.DataFrame(columns=gm_anthro_female.columns)

n_regions = 400
correlations_female = np.zeros(n_regions)
p_values_female = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_female.iloc[:, region], mri_anthro_female['predicted_hgs'])
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

n_regions = 400
correlations_male = np.zeros(n_regions)
p_values_male = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_male.iloc[:, region], mri_anthro_male['predicted_hgs'])
    correlations_male[region] = corr
    p_values_male[region] = p_value
gm_region_corr_predicted_male.loc['correlations',:]= correlations_male.tolist()
gm_region_corr_predicted_male.loc['p_values',:]= p_values_male.tolist()

# Print correlations and p-values for each region
for region in range(n_regions):
    print(f"{gm_anthro_male.columns[region]}: Correlation_male = {correlations_male[region]:.3f}, p-value_male = {p_values_male[region]:.3f}")

##############################################################################################################################################################
# Calculate correlations between gm and Actual HGS
# gm_region_corr_actual = pd.DataFrame(columns=gm_anthro.columns)

# n_regions = 400
# n_subs = len(gm_anthro)
# correlations = np.zeros(n_regions)
# p_values = np.zeros(n_regions)
# for region in range(n_regions):
#     corr, p_value = spearmanr(gm_anthro.iloc[:, region], mri_anthro['actual_hgs'])
#     correlations[region] = corr
#     p_values[region] = p_value

# gm_region_corr_actual.loc['correlations',:]= correlations.tolist()
# gm_region_corr_actual.loc['p_values',:]= p_values.tolist()

# # Print correlations and p-values for each region
# for region in range(n_regions):
#     print(f"{gm_anthro.columns[region]}: Correlation = {correlations[region]:.3f}, p-value = {p_values[region]:.3f}")
###############################################################################
# Females
# Calculate correlations between gm and Actual HGS
gm_region_corr_actual_female = pd.DataFrame(columns=gm_anthro_female.columns)

n_regions = 400
correlations_female = np.zeros(n_regions)
p_values_female = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_female.iloc[:, region], mri_anthro_female['actual_hgs'])
    correlations_female[region] = corr
    p_values_female[region] = p_value

gm_region_corr_actual_female.loc['correlations',:]= correlations_female.tolist()
gm_region_corr_actual_female.loc['p_values',:]= p_values_female.tolist()

# Print correlations and p-values for each region
for region in range(n_regions):
    print(f"{gm_anthro_female.columns[region]}: Correlation_female = {correlations_female[region]:.3f}, p-value_female = {p_values_female[region]:.3f}")
###############################################################################
# Males
# Calculate correlations between gm and Actual HGS
gm_region_corr_actual_male = pd.DataFrame(columns=gm_anthro_male.columns)

n_regions = 400
correlations_male = np.zeros(n_regions)
p_values_male = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = spearmanr(gm_anthro_male.iloc[:, region], mri_anthro_male['actual_hgs'])
    correlations_male[region] = corr
    p_values_male[region] = p_value
gm_region_corr_actual_male.loc['correlations',:]= correlations_male.tolist()
gm_region_corr_actual_male.loc['p_values',:]= p_values_male.tolist()

# Print correlations and p-values for each region
for region in range(n_regions):
    print(f"{gm_anthro_male.columns[region]}: Correlation_male = {correlations_male[region]:.3f}, p-value_male = {p_values_male[region]:.3f}")
##############################################################################################################################################################
# MALES
df_actual_male = gm_region_corr_actual_male.T
# cast the strings to floats
df_actual_male['correlations'] = abs(df_actual_male['correlations'].astype(float))
top10_actual_male = df_actual_male.nlargest(10, 'correlations')
highest_region_male = top10_actual_male.iloc[0].name

df_predicted_male = gm_region_corr_predicted_male.T
# cast the strings to floats
df_predicted_male['correlations'] = abs(df_predicted_male['correlations'].astype(float))
top10_predicted_male = df_predicted_male.nlargest(10, 'correlations')
highest_region_predicted_male = top10_predicted_male.iloc[0].name
###############################################################################
# Females   
df_actual_female = gm_region_corr_actual_female.T
# cast the strings to floats
df_actual_female['correlations'] = abs(df_actual_female['correlations'].astype(float))
top10_actual_female = df_actual_female.nlargest(10, 'correlations')
highest_region_female = top10_actual_female.iloc[0].name

df_predicted_female = gm_region_corr_predicted_female.T
# cast the strings to floats
df_predicted_female['correlations'] = abs(df_predicted_female['correlations'].astype(float))
top10_predicted_female = df_predicted_female.nlargest(10, 'correlations')
highest_region_predicted_female = top10_predicted_female.iloc[0].name

# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################################################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
# # With and Without confound
corr_actual_predicted_female, p_actual_predicted_female = spearmanr(abs(df_actual_female['correlations']), abs(df_predicted_female['correlations']))
corr_actual_predicted_male, p_actual_predicted_male = spearmanr(abs(df_actual_male['correlations']), abs(df_predicted_male['correlations']))

df_predicted_male = df_predicted_male.rename(columns={'correlations':'correlations_pred'})
df_predicted_female = df_predicted_female.rename(columns={'correlations':'correlations_pred'})

df_male = pd.concat([df_actual_male, df_predicted_male], axis=1)
df_male.loc[:, "Gender"] = "Males"
df_female = pd.concat([df_actual_female, df_predicted_female], axis=1)
df_female.loc[:, "Gender"] = "Females"

df = pd.concat([df_male, df_female], axis=0)
# print("===== Done! =====")
# embed(globals(), locals())
###################################################################################
# def merge_plot(background, layer_list):
#     from PIL import Image
#     bg = Image.open(background)
#     #merge the layers
#     for i in layer_list:
#         layer = Image.open(i)
#         bg.paste(layer, (0, 0), layer)
#     return bg

# ###################################################################################
# color_dict = {'Males':'Blue',
#               'Females':'Red'}
# cmap_dict = {'Males':'Blues',
#              'Females':'Reds'}

# list_gender = ['Males', 'Females']
# from PIL import Image

# for i in range(0, len(list_gender)):
#     if i == 0:
#         dataset = df[df['Gender']==list_gender[i]]
#         max_x_value = dataset['correlations'].max()
#         min_x_value = dataset['correlations'].min()
#         max_y_value = dataset['correlations_pred'].max()
#         min_y_value = dataset['correlations_pred'].min()
#     elif i>0:
#         dataset = df[df['Gender']==list_gender[i]]
#         if (dataset['correlations'].max() > max_x_value):
#             max_x_value = dataset['correlations'].max()
#         if (dataset['correlations'].min() < min_x_value):
#             min_x_value = dataset['correlations'].min()
#         if (dataset['correlations_pred'].max() > max_y_value):
#             max_y_value = dataset['correlations_pred'].max()
#         if (dataset['correlations_pred'].min() < min_y_value):
#             min_y_value = dataset['correlations_pred'].min()

# save_name_m = []
# sns.set_theme(style="ticks")
# ax = plt.figure(figsize=(14,12))
# for i in range(0, len(list_gender)):
#     if i == 0:
#         dataset = df[df['Gender']==list_gender[i]]
#         g = sns.jointplot(data=dataset, x='correlations', y='correlations_pred',
#                         alpha=0.9, 
#                         color=list(color_dict.values())[i],
#                         mincnt=1, kind='hex', gridsize=15, height=4, ratio=5)
#         xmin, xmax = g.ax_joint.get_xlim()
#         ymin, ymax = g.ax_joint.get_ylim()
#         max_axis = max(xmax, ymax)
#         min_axis = min(xmin, ymin)
#         # JointGrid has a convenience function
#         g.set_axis_labels('x', 'y', fontsize=10)

#         # or set labels via the axes objects
#         g.ax_joint.set_xlabel('Correlation of GM with Actual HGS', fontweight='bold')
#         g.ax_joint.set_ylabel('Correlation of GM with Predicted HGS', fontweight='bold')
#     elif i>0:
#         dataset = df[df['Gender']==list_gender[i]]
#         g = sns.jointplot(data=dataset, x='correlations', y='correlations_pred',
#                         alpha=0.9, 
#                         color=list(color_dict.values())[i],
#                         mincnt=1, kind='hex', gridsize=15, height=4, ratio=5)
#         xmin, xmax = g.ax_joint.get_xlim()
#         ymin, ymax = g.ax_joint.get_ylim()
#         if max(xmax, ymax) > max_axis:
#             max_axis = max(xmax, ymax)
#         if min(xmin, ymin) < min_axis:
#             min_axis = min(xmin, ymin)
            
#     g.ax_joint.set_xlim(min_axis, .4)
#     g.ax_joint.set_ylim(min_axis, .4)
#     g.ax_marg_x.set_xlim(min_axis, .4)
#     g.ax_marg_y.set_ylim(min_axis, .4)
#     # JointGrid has a convenience function
#     g.set_axis_labels('x', 'y', fontsize=10)

#     # or set labels via the axes objects
#     g.ax_joint.set_xlabel('Correlation of GM with Actual HGS', fontweight='bold')
#     g.ax_joint.set_ylabel('Correlation of GM with Predicted HGS', fontweight='bold')
#     if i==0:
#         j="Males"
#     if i==1:
#         j="Females"
#     s_name = 'hex_' + j +'.png'
#     save_name_m.append(s_name)
#     plt.savefig(s_name, transparent=True)
#     plt.show()

# #read all the plots, merge and save the result 
# h1m = Image.open(save_name_m[0])
# h2m = Image.open(save_name_m[1])
# h1m.paste(h2m, (0, 0), h2m)
# h1m.save('hex_tr.png')

# ### with marginal axes
# save_name_m = []
# sns.set_theme(style="ticks")
# ax = plt.figure(figsize=(14,12))
# for i in list_gender:
#     dataset = df[df['Gender']==i]
#     g=sns.JointGrid(data=dataset, x="correlations", y="correlations_pred", 
#                     height=4, ratio=5)
#     # g.ax_joint.hexbin(data=dataset, x="correlations", y="correlations_pred", 
#     #                 cmap="Blues", gridsize=15, alpha=.6)
#     g.plot_joint(sns.regplot, color="black", scatter=False)
#     # reg_line = plt.plot([0, .4], [0, .4], 'k--')
#     # g.plot_joint(sns.regplot, 'k--')
#     g.plot_marginals(sns.kdeplot, fill=True, color=color_dict.get(i))
    
#     g.ax_joint.set_xlim(min_axis, .4)
#     g.ax_joint.set_ylim(min_axis, .4)
#     g.ax_marg_x.set_xlim(min_axis,.4)
#     g.ax_marg_y.set_ylim(min_axis,.4)
    
#     # JointGrid has a convenience function
#     g.set_axis_labels('x', 'y', fontsize=10)

#     # or set labels via the axes objects
#     g.ax_joint.set_xlabel('Correlation of GM with Actual HGS', fontweight='bold')
#     g.ax_joint.set_ylabel('Correlation of GM with Predicted HGS', fontweight='bold')

#     s_name = 'hex_' + i +'.png'
#     save_name_m.append(s_name)
#     plt.savefig(s_name, transparent=True)
#     plt.show()

# #read all the plots, merge and save the result 
# h1m = Image.open(save_name_m[0])
# h2m = Image.open(save_name_m[1])
# h1m.paste(h2m, (0, 0), h2m)
# h1m.save('j_hex_tr.png')

# Image.open('hex_tr.png')
# Image.open('j_hex_tr.png')

# background = 'hex_tr.png'
# layer_list = ['j_hex_tr.png']
# bg1 = merge_plot(background, layer_list)
# bg1.save('bg_hex6.png')
# print("===== Done! =====")
# embed(globals(), locals())

##############################################################################################################################################################
fig, ax = plt.subplots(1, 2, figsize=(20,10))
plt.rcParams['font.weight'] = 'bold'

ax[0].set_box_aspect(1)
# Adjust the aspect ratio
sns.set_context("poster")
# sns.jointplot(x=df_actual_male['correlations'], y=df_predicted_male['correlations'], kind='hex', gridsize=20, cmap='Blues', ax=ax[0])
# plot_joint(sns.histplot, stat='density', bins=20, color='white', edgecolor='black', linewidth=0.5)
# ax[0].hexbin(x=df_actual_male['correlations'], y=df_predicted_male['correlations'], gridsize = 50, cmap ='Blues')
sns.regplot(x=df_actual_male['correlations'], y=df_predicted_male['correlations_pred'], ax=ax[0], line_kws={"color": "red"})
ax[0].tick_params(axis='both', labelsize=20)

ax[0].set(xlabel =None, ylabel=None)
xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

# max0 = max(xmax0, ymax0)
# min0 = min(xmin0, ymin0)
ax[0].set_ylim(xmin0, xmax0)
ax[0].set_xlim(ymin0, ymax0)
xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()
text0 = 'CORR: ' + str(format(corr_actual_predicted_male, '.3f'))
# ax[0].set_title(f"Males - Corr={format(corr_actual_predicted_male, '.3f')}", fontsize=15, fontweight="bold", y=1.02)
# ax[0].text(xmax0 - 0.05 * xmax0, ymax0 - 0.01 * ymax0, text0, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")
ax[0].set_title(f"Males(N={len(mri_anthro_male)})", fontsize=15, fontweight="bold", y=1)
# # Add a diagonal line
# ax[0].plot([ymin0, xmax0], [ymin0, ymax0], 'k--')

#################################
ax[1].set_box_aspect(1)
sns.set_context("poster")
# sns.jointplot(x=df_actual_female['correlations'], y=df_predicted_female['correlations'], kind='hex', gridsize=20, cmap='Oranges', ax=ax[1])
# ax[1].hexbin(x=df_actual_female['correlations'], y=df_predicted_female['correlations'], gridsize = 50, cmap ='Oranges')
sns.regplot(x=df_actual_female['correlations'], y=df_predicted_female['correlations_pred'], ax=ax[1], line_kws={"color": "red"}, color="orange")
ax[1].tick_params(axis='both', labelsize=20)

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()
# max1 = max(xmax1, ymax1)
# min1 = min(xmin1, ymin1)
ax[1].set_xlim(xmin1, xmax1)
ax[1].set_ylim(ymin1, ymax1)
xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()

text1 = 'CORR: ' + str(format(corr_actual_predicted_female, '.3f'))
# ax[1].set_title(f"Females - Corr={format(corr_actual_predicted_female, '.3f')}", fontsize=15, fontweight="bold", y=1.02)
ax[1].set_title(f"Females(N={len(mri_anthro_female)})", fontsize=15, fontweight="bold", y=1)
# ax[1].text(xmax1 - 0.05 * xmax1, ymax1 - 0.01 * ymax1, text1, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")
# ax[1].plot([0, xmax1], [0, ymax1], 'k--')

# Add a diagonal line
ax[1].set(xlabel =None, ylabel=None)


xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()
xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()

xmin = min(xmin0, xmin1)
xmax = max(xmax0, xmax1)
ymin = min(ymin0, ymin1)
ymax = max(ymax0, ymax1)


ax[0].set_xlim(xmin, xmax)
ax[0].set_ylim(ymin, ymax)

ax[1].set_xlim(xmin, xmax)
ax[1].set_ylim(ymin, ymax)

ax[0].plot([xmin, xmax], [ymin, ymax], 'k--')
ax[1].plot([xmin, xmax], [ymin, ymax], 'k--')

ax[0].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text0, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")
ax[1].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text1, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# # Specify the format of the tick labels
# ax[0].xaxis.set_major_formatter('{:.2f}')  # Format x-axis tick labels to display 2 decimal places
# ax[0].yaxis.set_major_formatter('{:.2f}')  # Format x-axis tick labels to display 2 decimal places

# # Specify the format of the tick labels
# ax[1].xaxis.set_major_formatter('{:.2f}')  # Format x-axis tick labels to display 2 decimal places
# ax[1].yaxis.set_major_formatter('{:.2f}')  # Format x-axis tick labels to display 2 decimal places

fig.text(0.5, 0.05, 'Correlation of GCOR with Actual HGS', ha='center', fontsize=20, fontweight="bold")
fig.text(0.05, 0.5, 'Correlation of GCOR with Predicted HGS', va='center',
        rotation='vertical', fontsize=20, fontweight="bold")
plt.suptitle("Correlation of GCOR with Actual HGS vs Predicted HGS", fontsize=20, fontweight="bold", y=.95)

plt.show()
plt.savefig(f"correlate_gcor_actual_predicted_hgs_stroke_1.png")
plt.close()
# print("===== Done! =====")
# embed(globals(), locals())
# ##############################################################################################################################################################
