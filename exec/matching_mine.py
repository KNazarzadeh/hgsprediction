
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

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ptpython.repl import embed
# # print("===== Done! =====")
# # embed(globals(), locals())

###############################################################################

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]


folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "results_hgsprediction",
        "healthy",
        "nonmri",
        "anthropometrics_age",
        f"{target}",
        "without_confound_removal",
        "data_ready_to_train_models",
        "both_gender",
        )

file_path = os.path.join(
        folder_path,
        "data_extracted_to_train_models.csv")
    
    # Load the dataframe from csv file path
df_nonmri = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
df_healthy = df_nonmri.copy()
df_healthy_female = df_nonmri[df_nonmri['gender']==0]
df_healthy_male = df_nonmri[df_nonmri['gender']==1]

# df_mri_1st_scan = healthy.load_hgs_predicted_results("healthy",
#     "mri",
#     "linear_svm",
#     "anthropometrics_age",
#     f"{target}",
#     "both_gender",
#     session="2",
# )
# df_healthy = df_mri_1st_scan[["gender", "1st_scan_age", "1st_scan_bmi",  "1st_scan_height",  "1st_scan_waist_to_hip_ratio", f"1st_scan_{target}"]]

# df_healthy.rename(columns={"1st_scan_age":"age", "1st_scan_bmi":"bmi",  "1st_scan_height":"height",  "1st_scan_waist_to_hip_ratio":"waist_to_hip_ratio",
#                            "1st_scan_handedness":"handedness", f"1st_scan_{target}":f"{target}"}, inplace=True)
# print(df_healthy)

# df_healthy_female = df_mri_1st_scan[df_mri_1st_scan['gender']==0]
# df_healthy_male = df_mri_1st_scan[df_mri_1st_scan['gender']==1]

###############################################################################

stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_stroke = stroke.load_hgs_predicted_results("stroke", mri_status, session_column, model_name, feature_type, target, "both_gender")
df_pre_stroke = df_stroke[["gender", "1st_pre-stroke_age", "1st_pre-stroke_bmi",  "1st_pre-stroke_height",  "1st_pre-stroke_waist_to_hip_ratio", f"1st_pre-stroke_{target}"]]
df_pre_stroke.rename(columns={"1st_pre-stroke_age":"age", "1st_pre-stroke_bmi":"bmi",  "1st_pre-stroke_height":"height",  "1st_pre-stroke_waist_to_hip_ratio":"waist_to_hip_ratio", 
                              "1st_pre-stroke_handedness":"handedness", f"1st_pre-stroke_{target}":f"{target}"}, inplace=True)

df_post_stroke = df_stroke[["gender", "1st_post-stroke_age", "1st_post-stroke_bmi",  "1st_post-stroke_height",  "1st_post-stroke_waist_to_hip_ratio", f"1st_post-stroke_{target}"]]
df_post_stroke.rename(columns={"1st_post-stroke_age":"age", "1st_post-stroke_bmi":"bmi",  "1st_post-stroke_height":"height",  "1st_post-stroke_waist_to_hip_ratio":"waist_to_hip_ratio",
                               "1st_post-stroke_handedness":"handedness", f"1st_post-stroke_{target}":f"{target}"}, inplace=True)

df_stroke_female = df_stroke[df_stroke['gender']==0]
df_stroke_male = df_stroke[df_stroke['gender']==1]

###############################################################################

df_pre_stroke["disease"] = 1
df_healthy["disease"] = 0

df_pre = pd.concat([df_healthy, df_pre_stroke])
df_pre['index'] = df_pre.index

df_post_stroke["disease"] = 1
df_healthy["disease"] = 0
df_post = pd.concat([df_healthy, df_post_stroke])
df_post['index'] = df_post.index

df_pre_female=df_pre[df_pre["gender"]==0]
df_pre_male=df_pre[df_pre["gender"]==1]
df_post_female=df_post[df_post["gender"]==0]
df_post_male=df_post[df_post["gender"]==1]
print("===== Done! =====")
embed(globals(), locals())

# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import pairwise_distances
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression

# # Assuming you have a DataFrame called "data" with covariates (e.g., age, BMI, height, waist-to-hip ratio) and a treatment indicator

# # Define your covariates (predictors)
# covariates = ['age','height']

# # Extract the covariates and the treatment indicator
# X = df_pre_female[covariates]
# treatment = df_pre_female['disease']

# # Fit a logistic regression model to predict treatment assignment
# model = LogisticRegression()
# model.fit(X, treatment)

# # Get the predicted probabilities (propensity scores)
# propensity_scores = model.predict_proba(X)[:, 1]

# # Add the propensity scores to your DataFrame
# df_pre_female['propensity_score'] = propensity_scores

# # Assuming you have a DataFrame called "data" with propensity scores and treatment indicator
# # Create separate DataFrames for treated and control groups
# treated_data = df_pre_female[df_pre_female['disease'] == 1]
# control_data = df_pre_female[df_pre_female['disease'] == 0]

# # Standardize the covariates (propensity scores) to have zero mean and unit variance
# scaler = StandardScaler()
# treated_covariates = scaler.fit_transform(treated_data[['propensity_score']])
# control_covariates = scaler.transform(control_data[['propensity_score']])

# # Find exact matches
# neigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean')
# neigh.fit(control_covariates)
# distances, indices = neigh.kneighbors(treated_covariates)

# # Get the matched pairs
# matched_treated = treated_data.loc[:, df_pre_female.columns != 'index']
# matched_control = control_data.iloc[indices.flatten()].loc[:, df_pre_female.columns != 'index']

# # Combine the matched pairs into a new DataFrame
# matched_data = pd.concat([matched_treated, matched_control], axis=0)

# # Reset the index
# matched_data = matched_data.reset_index(drop=True)

mydata = df_pre_female.copy()
mydata['Group'] = mydata['disease'] == 1
# Assuming you have a DataFrame named 'mydata' with columns 'Group', 'Age', and 'Sex'
# Separate data into treatment and control groups
treatment_group = mydata[mydata['Group'] == True]
control_group = mydata[mydata['Group'] == False]

# Calculate the distance matrix based on Age and Sex
distance_matrix = cdist(treatment_group[["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]], control_group[["age", "bmi",  "height",  "waist_to_hip_ratio",f"{target}"]], metric='euclidean')

# Perform nearest neighbor matching with a 1:1 ratio
row_indices, col_indices = linear_sum_assignment(distance_matrix)

# Create a DataFrame with matched pairs
matched_data = pd.concat([
    treatment_group.iloc[row_indices].reset_index(drop=True),
    control_group.iloc[col_indices].reset_index(drop=True)
], axis=1)

###############################################################################
# "bmi",  "height",  "waist_to_hip_ratio",  "hgs_L+R", "gender"
psm = PsmPy(df_pre_female, treatment='disease', indx='index', exclude=["age", "height", f"{target}"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
df_pre_matched_female = psm.df_matched

df_pre_matched_female_controls = psm.matched_ids["matched_ID"]
df_pre_matched_female_stroke = psm.matched_ids["index"]

df_healthy_pre_female = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_pre_matched_female_controls)]

###############################################################################

psm = PsmPy(df_pre_male, treatment='disease', indx='index', exclude=[f"{target}"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
df_pre_matched_male = psm.df_matched

df_pre_matched_male_controls = psm.matched_ids["matched_ID"]
df_pre_matched_male_stroke = psm.matched_ids["index"]
df_healthy_pre_male = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_pre_matched_male_controls)]
###############################################################################
psm = PsmPy(df_post_female, treatment='disease', indx='index', exclude=[f"{target}"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
df_post_matched_female = psm.df_matched

df_post_matched_female_controls = psm.matched_ids["matched_ID"]
df_post_matched_female_stroke = psm.matched_ids["index"]
df_healthy_post_female = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_post_matched_female_controls)]

###############################################################################
psm = PsmPy(df_post_male, treatment='disease', indx='index', exclude=[f"{target}"])
psm.logistic_ps(balance=False)
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
df_post_matched_male = psm.df_matched

df_post_matched_male_controls = psm.matched_ids["matched_ID"]
df_post_matched_male_stroke = psm.matched_ids["index"]
df_healthy_post_male = df_mri_1st_scan[df_mri_1st_scan.index.isin(df_post_matched_male_controls)]
###############################################################################
###############################################################################
fig, ax = plt.subplots(2,2, figsize=(16,10))

sns.histplot(df_pre_matched_female, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[0,0])
sns.histplot(df_post_matched_female, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[0,1])
sns.histplot(df_pre_matched_male, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[1,0])
sns.histplot(df_post_matched_male, x='propensity_logit', hue='disease', multiple="dodge", shrink=.8, ax=ax[1,1])

# Add titles to the subplots
ax[0, 0].set_title(f'Pre-Matched Female(N={len(df_pre_matched_female[df_pre_matched_female["disease"]==1])})')
ax[0, 1].set_title(f'Post-Matched Female(N={len(df_post_matched_female[df_post_matched_female["disease"]==1])})')
ax[1, 0].set_title(f'Pre-Matched Male(N={len(df_pre_matched_male[df_pre_matched_male["disease"]==1])})')
ax[1, 1].set_title(f'Post-Matched Male(N={len(df_post_matched_male[df_post_matched_male["disease"]==1])})')

# # Add custom legend labels
legend_labels = ['healthy', 'stroke']  # Replace with your desired labels

# Create custom legends for each subplot
for i in range(2):
    for j in range(2):
        ax[i, j].legend(labels=legend_labels)


plt.suptitle("Pre- and Post-stroke and healthy matched samples")
# Adjust layout to prevent title overlap
plt.tight_layout()

plt.show()
plt.savefig("matching.png")

print("===== Done! =====")
embed(globals(), locals())
