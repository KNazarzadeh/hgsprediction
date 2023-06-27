
import pingouin as pg
# from pingouin import partial_corr
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
from statsmodels.stats.multitest import fdrcorrection

from ptpython.repl import embed
 

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
df_name = ["1_post_session", "2_post_session", "3_post_session", "4_post_session"]

mri_status = "mri"
population = "stroke"

df_post = pd.DataFrame()
save_folder_path = os.path.join(
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

for i in range(0,4):
    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{df_name[i]}_{mri_status}_{population}.csv")
    
    df_tmp = pd.read_csv(save_file_path, sep=',')
    df_post = pd.concat([df_post, df_tmp], axis=0)
    df_post = df_post.drop_duplicates()   
##############################################################################
# Replace Age
df_post.loc[:, 'post_age'] = df_post.loc[:, f'21003-2.0']
##############################################################################
# Replace BMI
df_post.loc[:, 'post_bmi'] = df_post.loc[:, f'21001-2.0']
##############################################################################
# Replace Height
df_post.loc[:, 'post_height'] = df_post.loc[:, f'50-2.0']

df_post.loc[:, 'post_days'] = df_post.loc[:, f'followup_days-2.0']

##############################################################################
# Replace waist to hip ratio
df_post.loc[:, 'post_waist'] = df_post.loc[:, f'48-2.0']
df_post.loc[:, 'post_hip'] = df_post.loc[:, f'49-2.0']

df_post['post_waist_hip_ratio'] = (df_post.loc[:, "post_waist"].astype(str).astype(float)).div(
                df_post.loc[:, "post_hip"].astype(str).astype(float))
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
df_post.loc[:, f"post_hgs(L+R)"] = \
            df_post.loc[:, f"46-2.0"] + df_post.loc[:, f"47-2.0"]
            
df_post = df_post[df_post.loc[:, f"dominant_hgs"] >=4]

##############################################################################
# extract_features = ExtractFeatures(df_tmp_mri, motor, population)
# extracted_data = extract_features.extract_features()
# Remove columns that all values are NaN
nan_cols = df_post.columns[df_post.isna().all()].tolist()
df_test_set = df_post.drop(nan_cols, axis=1)

mri_features = df_test_set
# print("===== Done! =====")
# embed(globals(), locals())
# X = define_features(feature_type, new_data)
X = ["post_age", "post_bmi", "post_height", "post_waist_hip_ratio", '31-0.0', 'post_days']
# Target: HGS(L+R)
# y = define_target(target)
if target == "L+R":
    y = "post_hgs(L+R)"

###############################################################################
# Remove Missing data from Features and Target
mri_features = mri_features.dropna(subset=y)
mri_features = mri_features.dropna(subset=X)
mri_features = mri_features.set_index('SubjectID')
# print("===== Done! =====")
# embed(globals(), locals())
new_data = mri_features[X]
new_data = new_data.rename(columns={'post_age': 'Age1stVisit', 'post_bmi': '21001-0.0', 'post_height':'50-0.0', 'post_waist_hip_ratio': 'waist_to_hip_ratio-0.0'})
new_data = pd.concat([new_data, mri_features[y]], axis=1)

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

f_days = mri_features[mri_features['31-0.0']==0.0]['post_days']
f_hgs_LR = df_test_female["post_hgs(L+R)"]

m_days = mri_features[mri_features['31-0.0']==1.0]['post_days']
m_hgs_LR = df_test_male["post_hgs(L+R)"]
# print("===== Done! =====")
# embed(globals(), locals())
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
# print("===== Done! =====")
# embed(globals(), locals())

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
################################################################################
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
    

# ###############################################################################
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
    

df_female = pd.concat([gm_anthro_female,df_test_female['post_hgs(L+R)'], df_test_female['predicted_hgs']], axis=1)
df_pingouin_female = pd.DataFrame()
for i in range(0,gm_anthro_female.shape[1]):
    gm_region =  df_female.columns[i]
    ping_results = pg.partial_corr(data=df_female, x='post_hgs(L+R)', y=gm_region, x_covar='predicted_hgs', method='spearman')
    ping_results.index = [gm_region]
    df_pingouin_female = pd.concat([df_pingouin_female, ping_results], axis=0)

df_male = pd.concat([gm_anthro_male,df_test_male['post_hgs(L+R)'], df_test_male['predicted_hgs']], axis=1)
df_pingouin_male = pd.DataFrame()
for i in range(0,gm_anthro_male.shape[1]):
    gm_region =  df_male.columns[i]
    ping_results = pg.partial_corr(data=df_male, x='post_hgs(L+R)', y=gm_region, x_covar='predicted_hgs', method='spearman')
    ping_results.index = [gm_region]
    df_pingouin_male = pd.concat([df_pingouin_male, ping_results], axis=0)

print("===== Done! =====")
embed(globals(), locals())

# Example usage
p_values_female = df_pingouin_female['p-val']
rejected_female, p_values_adj_female = fdrcorrection(p_values_female, alpha=0.05, is_sorted=True)
df_pingouin_female.loc[:, 'fdr_p-val'] = p_values_adj_female

p_values_male = df_pingouin_male['p-val']
rejected_male, p_values_adj_male = fdrcorrection(p_values_male, alpha=0.05, is_sorted=True)
df_pingouin_male.loc[:, 'fdr_p-val'] = p_values_adj_male

ping_male = df_pingouin_male[['r', 'p-val','fdr_p-val']].sort_values(['r'], ascending=False)
top20_ping_male = ping_male.iloc[0:20, :]

ping_female = df_pingouin_female[['r', 'p-val', 'fdr_p-val']].sort_values(['r'], ascending=False)
top20_ping_female = ping_female.iloc[0:20, :]

###############################################################################
fig, ax = plt.subplots(1, 2, figsize=(30,10))
sns.set_context("poster")
ax[0].set_box_aspect(1)
sns.barplot(x=top20_ping_male.values, y=top20_ping_male.index, ax=ax[0], capsize=.4, errcolor=".5",
    linewidth=3, edgecolor=".5", facecolor='b')
ax[0].tick_params(axis='both', labelsize=20)

# xmin0, xmax0 = ax[0].get_xlim()
# ymin0, ymax0 = ax[0].get_ylim()

# # xmax0 = xmax0+2
# ymax0 = ymax0+5

# ax[0].set_xlim(0, xmax0)
# ax[0].set_ylim(0, ymax0)

# xmin0, xmax0 = ax[0].get_xlim()
# ymin0, ymax0 = ax[0].get_ylim()

# text0 = 'CORR: ' + str(format(corr_male_diff, '.4f'))
ax[0].set_xlabel('Partial correlation', fontsize=20, fontweight="bold")
# ax[0].set_ylabel('Difference (Actual-Predicted)HGS', fontsize=20, fontweight="bold")

ax[0].set_title(f"Males({len(df_test_male)})", fontsize=15, fontweight="bold", y=1)
# ax[0].text(xmax0 - 0.05 * xmax0, ymax0 - 0.01 * ymax0, text0, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
# ax[0].plot([xmin0, xmax0], [ymin0, ymax0], 'k--')

#################################
ax[1].set_box_aspect(1)
# sns.set_context("poster")
sns.barplot(x=top20_ping_female.values, y=top20_ping_female.index, ax=ax[1], capsize=.4, errcolor=".5",
    linewidth=3, edgecolor=".5", facecolor='red')
ax[1].tick_params(axis='both', labelsize=20)

# xmin1, xmax1 = ax[1].get_xlim()
# ymin1, ymax1 = ax[1].get_ylim()
# # xmax1 = xmax1+2
# # ymax1 = ymax1+5

# # ax[1].set_xlim(0, xmax1)
# # ax[1].set_ylim(0, ymax1)

# # xmin1, xmax1 = ax[1].get_xlim()
# # ymin1, ymax1 = ax[1].get_ylim()


# text1 = 'CORR: ' + str(format(corr_female_diff, '.4f'))
ax[1].set_title(f"Females({len(df_test_female)})", fontsize=15, fontweight="bold", y=1)


# # Add a diagonal line
ax[1].set_xlabel('Partial correlation', fontsize=20, fontweight="bold")
# ax[1].set_ylabel('GMV regions', fontsize=20, fontweight="bold")


# xmin0, xmax0 = ax[0].get_xlim()
# ymin0, ymax0 = ax[0].get_ylim()
# xmin1, xmax1 = ax[1].get_xlim()
# ymin1, ymax1 = ax[1].get_ylim()

# xmin = min(xmin0, xmin1)
# xmax = max(xmax0, xmax1)
# ymin = min(ymin0, ymin1)
# ymax = max(ymax0, ymax1)


# ax[0].set_xlim(xmin, xmax)
# ax[0].set_ylim(ymin, ymax)

# ax[1].set_xlim(xmin, xmax)
# ax[1].set_ylim(ymin, ymax)

# ax[0].plot([xmin, xmax], [ymin, ymax], 'k--')
# ax[1].plot([xmin, xmax], [ymin, ymax], 'k--')

# ax[0].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text0, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")
# ax[1].text(xmax - 0.05 * xmax, ymax - 0.01 * ymax, text1, verticalalignment='top',
#          horizontalalignment='right', fontsize=18, fontweight="bold")

plt.suptitle("Partial correlation between GMV regions and Actual (L+R)HGS - Top 20", fontsize=20, fontweight="bold", y=0.95)

plt.show()
plt.savefig(f"pingouin_corr_female_male.png")
plt.close()
