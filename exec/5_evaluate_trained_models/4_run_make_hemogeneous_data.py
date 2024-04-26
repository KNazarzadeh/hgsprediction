import math
import sys
import numpy as np
import pandas as pd
import os
from scipy.stats import zscore

from hgsprediction.define_features import define_features
from hgsprediction.load_results.load_corrected_prediction_results import load_corrected_prediction_results
from hgsprediction.save_results.save_zscore_results import save_zscore_results
from hgsprediction.load_results.load_zscore_results import load_zscore_results


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
###############################################################################
# Load z-score results for healthy individuals with MRI data
# And asssign to control dataframe
def load_control_session(session, gender):
    df_control_session = load_zscore_results(
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
    
    df_control_session.loc[:, "disorder"] = 0
    df_control_session.loc[:, "session"] = float(session)
    return df_control_session

# Load z-score results for healthy individuals with MRI data for each session
control_dataframes_male = []
control_dataframes_female = []
for gender in ["male", "female"]:
    for session_number in range(4):
        df_control_session = load_control_session(session_number, gender)
        if gender == "male":
            control_dataframes_male.append(df_control_session)
        elif gender == "female":
            control_dataframes_female.append(df_control_session)
            
# print("===== Done! End =====")
# embed(globals(), locals())
###############################################################################
for pre_ses in range(0, 4):
    df_pre = control_dataframes_female[pre_ses]
    for post_ses in range(pre_ses+1, 4):
        df_post = control_dataframes_female[post_ses]
        intersection_index = df_pre.index.intersection(df_post.index)
        df_control_pre_female = df_pre[df_pre.index.isin(intersection_index)]
        df_control_post_female = df_post[df_post.index.isin(intersection_index)]
        print(len(df_control_pre_female))


for pre_ses in range(0, 4):
    df_pre = control_dataframes_male[pre_ses]
    for post_ses in range(pre_ses+1, 4):
        df_post = control_dataframes_male[post_ses]
        intersection_index = df_pre.index.intersection(df_post.index)
        df_control_pre_male = df_pre[df_pre.index.isin(intersection_index)]
        df_control_post_male = df_post[df_post.index.isin(intersection_index)]
        print(len(df_control_pre_male))

min_size = min(len(df_control_pre_male), len(df_control_pre_female))

df_male_sample = df_control_pre_male.sample(n=min_size, random_state=42)
df_female_sample = df_control_pre_female.sample(n=min_size, random_state=42)

female_std = df_female_sample['hgs_L+R'].std()
# print("===== Done! End =====")
# embed(globals(), locals())
while True:
    male_std = df_control_pre_male['hgs_L+R'].std()
    # Break if standard deviations are close enough
    if np.isclose(male_std, female_std, atol=0.5):
        break

    # Adjust the group with the larger standard deviation by removing the furthest point from mean
    if male_std > female_std:
        male_mean = df_control_pre_male['hgs_L+R'].mean()
        furthest = df_control_pre_male['hgs_L+R'].sub(male_mean).abs().idxmax()
        df_male_sample = df_control_pre_male.drop(furthest)
    
    # Resample to maintain the size
    df_male_sample = df_control_pre_male.sample(n=min_size, random_state=42)
    
print("===== Done! End =====")
embed(globals(), locals())
# # Assuming DataFrame 'data' with columns 'gender' and 'hgs' exists
# male_data = df[df['gender'] == 1]
# female_data = df[df['gender'] == 0]

# # Divide the populations into strata based on HGS quintiles
# male_data['hgs_quintile'] = pd.qcut(male_data['hgs_L+R'], 50, labels=False, duplicates='drop')
# female_data['hgs_quintile'] = pd.qcut(female_data['hgs_L+R'], 50, labels=False, duplicates='drop')


# from sklearn.model_selection import StratifiedShuffleSplit

# # Adjust the test_size to 0.1, which implies that the training set will be 90%.
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=47)

# # Sampling for male and female data
# male_sample = pd.DataFrame()
# female_sample = pd.DataFrame()

# # Sampling 90% of males
# for train_idx, _ in split.split(male_data, male_data['hgs_quintile']):
#     male_train = male_data.iloc[train_idx]
#     male_strata = male_train.groupby('hgs_quintile')
#     for stratum, group in male_strata:
#         # Here you can use all the data in the training split, or further sample within it if needed
#         male_sample = pd.concat([male_sample, group])

# # Sampling 90% of females
# for train_idx, _ in split.split(female_data, female_data['hgs_quintile']):
#     female_train = female_data.iloc[train_idx]
#     female_strata = female_train.groupby('hgs_quintile')
#     for stratum, group in female_strata:
#         # Similar sampling as for males
#         female_sample = pd.concat([female_sample, group])


# from scipy.stats import levene
# levene(female_sample['hgs_L+R'], male_sample['hgs_L+R'])

print("===== Done! End =====")
embed(globals(), locals())

save_zscore_results(
    df,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)

print("===== Done! End =====")
embed(globals(), locals())