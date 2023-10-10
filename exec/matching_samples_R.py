
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
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]


def matching_samples(df, list_of_features):
    
    treatment_group = df[df['disease'] == 1]
    control_group = df[df['disease'] == 0]

    # Calculate the distance matrix based on Age and Sex
    distance_matrix = cdist(treatment_group[list_of_features], control_group[list_of_features], metric='euclidean')

    # Perform nearest neighbor matching with a 1:1 ratio
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Create a DataFrame with matched pairs
    df_matched_samples = pd.concat([
        treatment_group.iloc[row_indices].reset_index(drop=True),
        control_group.iloc[col_indices].reset_index(drop=True)], axis=1)
    
    return df_matched_samples


df_mri_1st_scan = healthy.load_hgs_predicted_results("healthy",
    "mri",
    "linear_svm",
    "anthropometrics_age",
    f"{target}",
    "both_gender",
    session="2",
)
df_healthy = df_mri_1st_scan[["gender", "1st_scan_age", "1st_scan_bmi",  "1st_scan_height",  "1st_scan_waist_to_hip_ratio", f"1st_scan_{target}"]]

df_healthy.rename(columns={"1st_scan_age":"age", "1st_scan_bmi":"bmi",  "1st_scan_height":"height",  "1st_scan_waist_to_hip_ratio":"waist_to_hip_ratio",
                           "1st_scan_handedness":"handedness", f"1st_scan_{target}":f"{target}"}, inplace=True)
print(df_healthy)

df_healthy_female = df_mri_1st_scan[df_mri_1st_scan['gender']==0]
df_healthy_male = df_mri_1st_scan[df_mri_1st_scan['gender']==1]

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
# df_pre['index'] = df_pre.index
df_pre.insert(0, "index", df_pre.index)
 
df_post_stroke["disease"] = 1
df_healthy["disease"] = 0
df_post = pd.concat([df_healthy, df_post_stroke])
# df_post['index'] = df_post.index
df_post.insert(0, "index", df_post.index)

df_pre_female=df_pre[df_pre["gender"]==0]
df_pre_male=df_pre[df_pre["gender"]==1]
df_post_female=df_post[df_post["gender"]==0]
df_post_male=df_post[df_post["gender"]==1]

print("===== Done! =====")
embed(globals(), locals())
##############################################################################
###############################################################################
treated_df = df_pre_female[df_pre_female['disease']==1]
non_treated_df = df_pre_female[df_pre_female['disease']==0]

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
def get_matching_pairs(treated_df, non_treated_df, scaler=True):
    treated_x = treated_df.values
    non_treated_x = non_treated_df.values
    if scaler == True:
        scaler = StandardScaler()
    if scaler:
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(non_treated_x)
    distances, indices = nbrs.kneighbors(treated_x)
    indices = indices.reshape(indices.shape[0])
    matched = non_treated_df.iloc[indices]
    return matched

##############################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Define the covariates you want to use for matching
# covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]
covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]

# Create a DataFrame for treated and control groups
treated_group = df_pre_female[df_pre_female['disease'] == 1]
control_group = df_pre_female[df_pre_female['disease'] == 0]

# Fit a Nearest Neighbors model on the control group
knn = NearestNeighbors(n_neighbors=1)
knn.fit(control_group[covariates])

# Find the nearest neighbors for each treated unit
distances, indices = knn.kneighbors(treated_group[covariates])

# Fit a logistic regression model to estimate propensity scores
propensity_model = LogisticRegression()
propensity_model.fit(df_pre_female[covariates], df_pre_female['disease'])
propensity_scores = propensity_model.predict_proba(df_pre_female[covariates])[:, 1]

def Match(groups, propensity, caliper = 0.05):
    ''' 
    Inputs:
    groups = Treatment assignments.  Must be 2 groups
    propensity = Propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    caliper = Maximum difference in matched propensity scores. For now, this is a caliper on the raw
            propensity; Austin reccommends using a caliper on the logit propensity.
    
    Output:
    A series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match.
    '''

    # Check inputs
    if any(propensity <=0) or any(propensity >=1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not(0<caliper<1):
        raise ValueError('Caliper must be between 0 and 1')
    elif len(groups)!= len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups')
        
        
    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups.sum(); N2 = N-N1
    g1, g2 = propensity[groups == 1], (propensity[groups == 0])
    # Check if treatment groups got flipped - treatment (coded 1) should be the smaller
    if N1 > N2:
       N1, N2, g1, g2 = N2, N1, g2, g1 
        
        
    # Randomly permute the smaller group to get order for matching
    morder = np.random.permutation(N1)
    matches = pd.Series(np.empty(N1))
    matches[:] = np.NAN
    
    for m in morder:
        dist = abs(g1[m] - g2)
        if dist.min() <= caliper:
            matches[m] = dist.argmin()
            g2 = g2.drop(matches[m])
    return (matches)


# Create a DataFrame to store the matched pairs with index columns and distances
matched_pairs = pd.DataFrame({
    'treated_index': treated_group.index,
    'control_index': control_group.index[indices.flatten()],
    'distance': distances.flatten(),
    'propensity_score': propensity_scores[indices.flatten()]
})

# Use the matched pairs to create the matched data
matched_data = treated_group.reset_index(drop=True).join(control_group.iloc[indices.flatten()].reset_index(drop=True), lsuffix="_treat", rsuffix="_control")

matched_data['distance'] = matched_pairs['distance'].values
matched_data['propensity_score'] = matched_pairs['propensity_score'].values


matched_treated = matched_data['propensity_score']
matched_controls = matched_data['propensity_score']
unmatched_units= df_pre_female

###############################################################################
mydata = df_pre_female.copy()
mydata['Group'] = mydata['disease'] == 1
# Assuming you have a DataFrame named 'mydata' with columns 'Group', 'Age', and 'Sex'
# Separate data into treatment and control groups
treatment_group = mydata[mydata['Group'] == True]
control_group = mydata[mydata['Group'] == False]

# Set a random seed for reproducibility
# Calculate the distance matrix based on Age and Sex
distance_matrix = cdist(treatment_group[["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]], control_group[["age", "bmi",  "height",  "waist_to_hip_ratio",f"{target}"]], metric='euclidean')

# Perform nearest neighbor matching with a 1:1 ratio
row_indices, col_indices = linear_sum_assignment(distance_matrix)

# Create a DataFrame with matched pairs
matched_data = pd.concat([
    treatment_group.iloc[row_indices].reset_index(drop=True),
    control_group.iloc[col_indices].reset_index(drop=True)], axis=1)

###############################################################################
###############################################################################
# Perform matching based on propensity scores
def match(data, treatment_column, propensity_score_column):
    treated = data[data[treatment_column] == 1]
    control = data[data[treatment_column] == 0]
    
    matched_control = control.sample(n=len(treated), weights=1 / (1 - control[propensity_score_column]))
    
    matched_data = pd.concat([treated, matched_control])
    
    return matched_data

###############################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming you have a DataFrame or relevant data from match.it
# You might need to replace this with your actual data
data = pd.DataFrame({
    'Treatment': np.random.rand(100),  # Replace with your treatment data
    'Control': np.random.rand(100),  # Replace with your control data
})

# Create a jitter plot using matplotlib
data=matched_data
plt.figure(figsize=(8, 6))
plt.scatter(data['Treatment'], np.zeros(len(data['Treatment'])), label='Treatment', alpha=0.5)
plt.scatter(data['Control'], np.ones(len(data['Control'])), label='Control', alpha=0.5)
plt.xlabel('Propensity Score')
plt.yticks([0, 1], ['Treatment', 'Control'])
plt.legend()
plt.title('Jitter Plot of Propensity Scores')
plt.grid(axis='y')
plt.show()
plt.savefig("gg.png")
plt.close()

