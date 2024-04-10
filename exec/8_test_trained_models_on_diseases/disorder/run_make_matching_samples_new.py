import math
import sys
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from hgsprediction.load_results.load_zscore_results import load_zscore_results
from hgsprediction.load_results.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results
from hgsprediction.define_features import define_features
from hgsprediction.save_results.save_disorder_matched_samples_results import save_disorder_matched_samples_results
from hgsprediction.save_results.save_disorder_matched_samples_correlation_results import save_disorder_matched_control_samples_correlation_results

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
disorder_cohort = sys.argv[9]
visit_session = sys.argv[10]
gender = sys.argv[11]
n_samples = sys.argv[12]

##############################################################################
# Load z-score results for healthy individuals with MRI data
# And asssign to control dataframe
session = 2
df_control_sessin_2 = load_zscore_results(
    "healthy",
    "mri",
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)

session = 3
# Load z-score results for healthy individuals with MRI data
# And asssign to control dataframe
df_control_sessin_3 = load_zscore_results(
    "healthy",
    "mri",
    model_name,
    feature_type,
    target,
    gender,
    "3",
    confound_status,
    n_repeats,
    n_folds,
)

intersection_index = df_control_sessin_2.index.intersection(df_control_sessin_3.index)
df_control_pre = df_control_sessin_2[df_control_sessin_2.index.isin(intersection_index)]
df_control_post = df_control_sessin_3[df_control_sessin_3.index.isin(intersection_index)]

# Reindex the dataframes to have the same order of indices
df_control_pre = df_control_pre.reindex(index=intersection_index)
df_control_post = df_control_post.reindex(index=intersection_index)

df_control_pre.loc[:, "disorder"] = 0
df_control_post.loc[:, "disorder"] = 0

# Check if the indices are in the same order
if df_control_pre.index.equals(df_control_post.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")


###############################################################################
features, extend_features = define_features(feature_type)

# Define features and target for matching
X = features
y = "disorder"

extract_columns = X + [y]

disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

df_disorder = load_disorder_corrected_prediction_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
)

df_disorder.loc[:, "disorder"] = 1

###############################################################################
df_control_pre_matched = pd.DataFrame()

for disorder_subgroup in [f"pre-{population}"]:

    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}_"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}_"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}_"
        
    if disorder_subgroup == f"pre-{population}":
        df_extracted = df_disorder[[col for col in df_disorder.columns if f"post-{population}" not in col]]

    elif disorder_subgroup == f"post-{population}":
        df_extracted = df_disorder[[col for col in df_disorder.columns if f"pre-{population}" not in col]]
    
    features_columns = [col for col in df_extracted.columns for item in extract_columns if item in col]
    
    # Remove the prefix from selected column names
    for col in features_columns:
        new_col_name = col.replace(prefix, "")
        df_extracted.rename(columns={col: new_col_name}, inplace=True)

    df = pd.concat([df_control_pre.loc[:, extract_columns], df_extracted.loc[:, extract_columns]], axis=0)

    # Initialize logistic regression model
    propensity_model = LogisticRegression()

    propensity_model.fit(df.loc[:, X], df.loc[:, y])
    # Prediction
    # probabilities for classes
    propensity_scores = propensity_model.predict_proba(df.loc[:, X])
    # the propensity score is the probability of being 1 (i.e., in the disorder group)
    df.loc[:, "propensity_scores"] = propensity_scores[:, 1]
    
    df_control_pre_tmp = df[df['disorder'] == 0]
    df_disorder_tmp = df[df['disorder'] == 1]
    
    df_disorder.loc[:, f"{prefix}propensity_scores"] = df_disorder_tmp.loc[:, "propensity_scores"]
    
    # caliper = np.std(df.loc[:, "propensity_scores"]) * 0.25

    # Fit nearest neighbors model on control_pre group using propensity scores
    nn_model = NearestNeighbors(n_neighbors=int(n_samples), algorithm='auto')
    nn_model.fit(df_control_pre_tmp['propensity_scores'].values.reshape(-1, 1))

    # Dictionary to store matched samples for each subject
    matched_samples = {}
    df_matched_tmp = pd.DataFrame()

    # Iterate over each row in treatment dataframe
    for subject_id, row in df_disorder_tmp.iterrows():
        propensity_score = row['propensity_scores']
        # Find 10 nearest neighbors for each treatment subject based on propensity scores
        distances, indices = nn_model.kneighbors([[propensity_score]])
        # Extract matched control_pre subjects
        matches = df_control_pre_tmp.iloc[indices[0]].index.tolist()
        matched_samples[subject_id] = matches

        df_matched_tmp = pd.concat([df_matched_tmp, df_control_pre[df_control_pre.index.isin(matches)]], axis=0)
        df_matched_tmp.loc[matches, "propensity_scores"] = df_control_pre_tmp[df_control_pre_tmp.index.isin(matches)].loc[:, "propensity_scores"]
        
        # Remove the matched subject from control_pre_tmp DataFrame to ensure it's not chosen again
        # df_control_pre_tmp.drop(df_matched_tmp.index, inplace=True)
        # print(len(df_control_pre_tmp))

    df_matched_tmp.loc[:, "disorder_episode"] = disorder_subgroup
    
    df_control_pre_matched = pd.concat([df_control_pre_matched, df_matched_tmp], axis=0)
        
    # Print matched samples for each subject
    for subject_id, matches in matched_samples.items():
        print(f"SubjectID: {subject_id}, Matches: {matches}")

# Add prefix to column names
df_control_pre_matched.columns = [prefix + col if col != 'gender' else col for col in df_control_pre_matched.columns]

disorder_subgroup = f"post-{population}"
prefix = f"1st_{disorder_subgroup}_"
df_control_post_matched = pd.DataFrame()
df_control_post_matched = df_control_post[df_control_post.index.isin(df_control_pre_matched.index)]
df_control_post_matched["disorder_episode"] = f"post-{population}"

# Reindex the dataframes to have the same order of indices
df_control_post_matched = df_control_post_matched.reindex(index=df_control_pre_matched.index)

# Check if the indices are in the same order
if df_control_pre_matched.index.equals(df_control_post_matched.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")

# Assuming prefix is the prefix you want to add and gender_column is the name of the gender column
# Add the prefix to the column names excluding the gender column(s)
df_control_post_matched.columns = [prefix + col if col != 'gender' else col for col in df_control_post_matched.columns]


df_control_matched = pd.concat([df_control_pre_matched, df_control_post_matched], axis=1)

# Check if the indices are in the same order
if df_control_matched.index.equals(df_control_pre_matched.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")

##############################################################################
sample_session = 0
save_disorder_matched_samples_results(
    df_control_matched,
    df_disorder,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
    sample_session,
    
)

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
df_correlations = pd.DataFrame(index=[f"pre-{population}", f"post-{population}"])
df_p_values = pd.DataFrame(index=[f"pre-{population}", f"post-{population}"])
df_r2_values = pd.DataFrame(index=[f"pre-{population}", f"post-{population}"])

for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:

    df = df_control_matched[df_control_matched["disorder_episode"] == disorder_subgroup]

    df_correlations.loc[disorder_subgroup, "r_values_true_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])[0]
    df_correlations.loc[disorder_subgroup, "r_values_true_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])[0]
    df_correlations.loc[disorder_subgroup, "r_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])[0]
    df_correlations.loc[disorder_subgroup, "r_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])[0]

    df_p_values.loc[disorder_subgroup, "r_values_true_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])[1]
    df_p_values.loc[disorder_subgroup, "r_values_true_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])[1]
    df_p_values.loc[disorder_subgroup, "r_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])[1]
    df_p_values.loc[disorder_subgroup, "r_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])[1]


    df_r2_values.loc[disorder_subgroup, "r2_values_true_predicted"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])
    df_r2_values.loc[disorder_subgroup, "r2_values_true_delta"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])
    df_r2_values.loc[disorder_subgroup, "r2_values_true_corrected_predicted"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])
    df_r2_values.loc[disorder_subgroup, "r2_values_true_corrected_delta"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])

save_disorder_matched_control_samples_correlation_results(
    df_correlations,
    df_p_values,
    df_r2_values,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
)

print("===== Done! End =====")
embed(globals(), locals())


