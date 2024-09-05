import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hgsprediction.define_features import define_features
from hgsprediction.load_results.disorder.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results
from hgsprediction.load_results.healthy.load_zscore_results import load_zscore_results
from hgsprediction.save_results.disorder.save_disorder_matched_samples_results import save_disorder_matched_samples_results

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
first_event = sys.argv[13]
##############################################################################
# Define features and target for matching
features, extend_features = define_features(feature_type)

# X = features
X_pre = ['pre_age', 'pre_bmi', 'pre_height', 'pre_waist_to_hip_ratio']
X_post = ['post_age']
y = "disorder"

extract_columns = X_pre + X_post + [y] + [target]
###############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
###############################################################################
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
    first_event,
)
# Assign 'disorder' column with a value of 1 (indicating disorder/disease)    
df_disorder.loc[:, "disorder"] = 1
###############################################################################
########################## ***** Load MRI Data ***** ##########################
###############################################################################
# Initialize an empty list to store dataframes for each session
control_dataframes = []
# Loop through session numbers 0 to 3
for session in ["0", "1", "2", "3"]:
    # Load the z-score results for the current session
    df_control_by_session = load_zscore_results(
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
    # Assign 'disorder' column with a value of 0 (indicating healthy control)    
    df_control_by_session.loc[:, "disorder"] = 0
    # Assign 'session' column with the current session number (converted to float)    
    df_control_by_session.loc[:, "session"] = float(session)
    # Append the dataframe for the current session to the list    
    control_dataframes.append(df_control_by_session)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Define prefix for pre- and post- columns
disorder_pre_subgroup = f"pre-{population}"
pre_prefix = "pre_"

disorder_post_subgroup = f"post-{population}"
# if visit_session == "1":
post_prefix = "post_"

# Rename columns by removing "1st_" and "-disorder"
df_disorder.columns = df_disorder.columns.str.replace('1st_', '').str.replace(f'-{population}', '')
###############################################################################
# Step 1: Calculate propensity scores for each patient
# Assuming you have a column named 'propensity_scores' in patient_df representing propensity scores
df_control_before_matched = []
df_control_matched = pd.DataFrame()

pre_ses_min = int(df_disorder[f"{pre_prefix}session"].min())
pre_ses_max = int(df_disorder[f"{pre_prefix}session"].max())

for pre_ses in range(pre_ses_min, pre_ses_max+1):
    print("pre_ses=", pre_ses)
    # Iterate over the range of session values
    df_disorder_pre_sessions = df_disorder[df_disorder[f"{pre_prefix}session"]==pre_ses]
    # Assuming df is your DataFrame
    if not df_disorder_pre_sessions.empty:
        df_control_pre = control_dataframes[pre_ses].copy()
        df_control_matched_tmp = pd.DataFrame()
        post_ses_min = int(df_disorder_pre_sessions[f"{post_prefix}session"].min())
        post_ses_max = int(df_disorder_pre_sessions[f"{post_prefix}session"].max())
        for post_ses in range(post_ses_min, post_ses_max+1):
            print("post_ses=", post_ses)
            # Iterate over the range of session values
            df_disorder_post_sessions = df_disorder[df_disorder[f"{post_prefix}session"]==post_ses]
            # Assuming df is your DataFrame
            if not df_disorder_post_sessions.empty:
                df_control_post = control_dataframes[post_ses].copy()
                intersection_index = df_control_pre.index.intersection(df_control_post.index)
                df_control_pre = df_control_pre[df_control_pre.index.isin(intersection_index)]
                df_control_post = df_control_post[df_control_post.index.isin(intersection_index)]

                # Reindex the dataframes to have the same order of indices
                df_control_pre = df_control_pre.reindex(index=intersection_index)
                df_control_post = df_control_post.reindex(index=intersection_index)

                # Check if the indices are in the same order
                if df_control_pre.index.equals(df_control_post.index):
                    print("The indices are in the same order.")
                else:
                    print("The indices are not in the same order.")

                df_control_pre.columns = ['pre_' + col if (col != 'disorder') and (col != 'gender') else col for col in df_control_pre.columns]
                
                df_control_post.columns = ['post_' + col if (col != 'disorder') and (col != 'gender') else col for col in df_control_post.columns]
                # print("===== Done! =====")
                # embed(globals(), locals())
                # Adding column 'post_age' from df_control_post to df_control_pre based on the same indexes
                df_control_extracted = df_control_pre[X_pre].join(df_control_post[X_post+['disorder']])
                
                df_disorder_extract = df_disorder_pre_sessions[df_disorder_pre_sessions.index.isin(df_disorder_post_sessions.index)]
                df_disorder_extract = df_disorder_extract[X_pre + X_post + ['disorder']]
                
                # df_disorder_pre = df_disorder_extract[[col for col in df_disorder_extract.columns if f"post-{population}" not in col]].copy()

                # features_columns = [col for col in df_disorder_pre.columns for item in extract_columns if item in col]

                # # Remove the prefix from selected column names
                # for col in features_columns:
                #     new_col_name = col.replace(pre_prefix, "")
                #     df_disorder_pre.rename(columns={col: new_col_name}, inplace=True)
        
                df = pd.concat([df_control_extracted, df_disorder_extract], axis=0)
                # print("===== Done! =====")
                # embed(globals(), locals())
                ###############################################################################
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('logistic_classifier', LogisticRegression())
                ])
                
                # Initialize logistic regression model
                X = X_pre + X_post
                propensity_model = pipe.fit(df.loc[:, X], df.loc[:, y])

                # Prediction
                # probabilities for classes
                propensity_scores = propensity_model.predict_proba(df.loc[:, X])
                # the propensity score is the probability of being 1 (i.e., in the disorder group)
                df.loc[:, "propensity_scores"] = propensity_scores[:, 1]
                ###############################################################################
                df_control_pre_tmp = df[df['disorder'] == 0].copy()
                df_disorder_tmp = df[df['disorder'] == 1].copy()
                ###############################################################################
                df_disorder.loc[df_disorder_extract.index, f"{pre_prefix}propensity_scores"] = df_disorder_tmp.loc[:, "propensity_scores"]
                ###############################################################################
                # Dictionary to store matched samples for each subject
                matched_samples = {}
                df_matched = pd.DataFrame()
                df_control_pre_matched = pd.DataFrame()
                df_control_post_matched = pd.DataFrame()
                # Iterate over each row in group dataframe
                for subject_id, row in df_disorder_tmp.iterrows():
                    df_matched_tmp = pd.DataFrame()
                    propensity_score = row['propensity_scores']
                    # Fit nearest neighbors model on control_pre group using propensity scores
                    caliper = np.std(df_control_pre_tmp.propensity_scores) * 0.05
                    nn_model = NearestNeighbors(n_neighbors=int(n_samples),radius=caliper)
                    nn_model.fit(df_control_pre_tmp['propensity_scores'].values.reshape(-1, 1))
                    
                    df_control_before_matched.append(df_control_pre_tmp)

                    # Find 10 nearest neighbors for each group subject based on propensity scores
                    distances, indices = nn_model.kneighbors([[propensity_score]])
                    print(indices)                
                    # Extract matched control_pre subjects
                    matches = df_control_pre_tmp.iloc[indices[0]].index.tolist()                
                    matched_samples[subject_id] = matches
                    print(matches)
                    df_matched_tmp = pd.concat([df_matched_tmp, df_control_pre[df_control_pre.index.isin(matches)]], axis=0)
                    df_matched_tmp = df_matched_tmp.reindex(matches)
                    df_matched_tmp.loc[matches, "propensity_scores"] = df_control_pre_tmp[df_control_pre_tmp.index.isin(matches)].loc[:, "propensity_scores"]
                    df_matched_tmp.loc[:, "patient_id"] = subject_id
                    
                    df_matched = pd.concat([df_matched, df_matched_tmp], axis=0)
                    df_control_pre_tmp.drop(index=matches, inplace=True)
                    
                df_matched.loc[:, "time_point"] = disorder_pre_subgroup

                df_control_pre_matched = pd.concat([df_control_pre_matched, df_matched], axis=0)

                # Print matched samples for each subject
                for subject_id, matches in matched_samples.items():
                    print(f"SubjectID: {subject_id}, Matches: {matches}")
                
                if df_matched[df_matched.index.duplicated()].empty:
                    print("There is no duplicate match.")
                else:
                    print("There is duplicate match:", df_matched[df_matched.index.duplicated()].index)
                
                ###############################################################################
                df_control_post_matched = df_control_post[df_control_post.index.isin(df_control_pre_matched.index)].copy()
                df_control_post_matched.loc[:, "time_point"] = f"post-{population}"
                # Reindex the dataframes to have the same order of indices
                df_control_post_matched = df_control_post_matched.reindex(index=df_control_pre_matched.index)
                df_control_post_matched.loc[:, "patient_id"] = df_control_pre_matched.loc[:, "patient_id"].astype(int)

                # Check if the indices are in the same order
                if df_control_pre_matched.index.equals(df_control_post_matched.index):
                    print("The indices are in the same order.")
                else:
                    print("The indices are not in the same order.")    
                # print("===== END Done End! =====")
                # embed(globals(), locals())
                ###############################################################################
                # Add prefix to column names
                # df_control_pre_matched.columns = [pre_prefix + col if col != 'gender' else col for col in df_control_pre_matched.columns]
                # # Add the prefix to the column names excluding the gender column(s)
                # df_control_post_matched.columns = [post_prefix + col if col != 'gender' else col for col in df_control_post_matched.columns]
                
                # Add 'pre_' to the specific column 'A'
                df_control_pre_matched = df_control_pre_matched.rename(columns={'patient_id': 'pre_patient_id'})
                # Add 'pre_' to the specific column 'A'
                df_control_post_matched = df_control_post_matched.rename(columns={'patient_id': 'post_patient_id'})
                
                df_control_matched_pre_post = pd.concat([df_control_pre_matched, df_control_post_matched], axis=1)
                # Drop duplicate columns by keeping the first occurrence
                df_control_matched_pre_post = df_control_matched_pre_post.loc[:, ~df_control_matched_pre_post.columns.duplicated()]
                
                # Check if the indices are in the same order
                if df_control_matched_pre_post.index.equals(df_control_pre_matched.index):
                    print("The indices are in the same order.")
                else:
                    print("The indices are not in the same order.")
                # print("===== END Done End! =====")
                # embed(globals(), locals())
                # Remove the specified suffixes from the column names in df1
                df_control_matched_pre_post.columns = df_control_matched_pre_post.columns.str.replace(r'-[0-3]\.0$', '', regex=True)

                df_control_matched_tmp = pd.concat([df_control_matched_tmp, df_control_matched_pre_post], axis=0)
                
                df_control_pre.drop(index=df_control_matched_tmp.index, inplace=True, errors='ignore')
                # Remove 'pre_' from column names if they contain it at the beginning
                df_control_pre.columns = df_control_pre.columns.str.replace('^pre_', '', regex=True)
                print(len(df_control_pre))
                print(pre_ses)
                print(len(df_control_post))
                print(post_ses)
                def remove_indices_from_dataframes(dataframes_list, indices_list):
                    for i in range(len(dataframes_list)):
                        df = dataframes_list[i]
                        df.drop(indices_list, axis=0, inplace=True, errors='ignore')
                remove_indices_from_dataframes(control_dataframes, df_control_matched_tmp.index.to_list())    
    
    df_control_matched = pd.concat([df_control_matched, df_control_matched_tmp], axis=0)
# print("===== END Done End! =====")
# embed(globals(), locals())
df_control_matched = df_control_matched[~df_control_matched.index.duplicated()]
not_same_values = df_control_matched[df_control_matched['pre_patient_id'] != df_control_matched['post_patient_id']]

if not_same_values.empty:
    print("pre and post controls are for the same paitent id")

print(df_control_matched)
print(df_disorder)
if df_control_matched[df_control_matched.index.duplicated()].empty:
    print("No Duplicated controls")

print("disorder_pre_age=", df_disorder['pre_age'].mean())

print("control_pre_age=", df_control_matched['pre_age'].mean())

print("disorder_post_age=", df_disorder['post_age'].mean())

print("control_post_age=", df_control_matched['post_age'].mean())
print("===== END Done End! =====")
embed(globals(), locals())
##############################################################################
###############################################################################
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
    first_event,
)
print("===== END Done End! =====")
embed(globals(), locals())
###############################################################################
