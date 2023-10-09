
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
    control_group.iloc[col_indices].reset_index(drop=True)
], axis=1)


print("===== Done! =====")
embed(globals(), locals())
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Assuming you have a dataframe 'lalonde' containing the data
# and you want to match on the specified variables

# Define the treatment and covariates
treatment = 'disease'
covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}", "gender"]

# Create a logistic regression model to estimate propensity scores
X = df_pre_female[covariates]
# X = sm.add_constant(X)  # Add a constant term for the intercept
y = df_pre_female[treatment]

logit_model = sm.Logit(y, X)
propensity_scores = logit_model.fit().predict()

# Add propensity scores to the original dataframe
lalonde['propensity_score'] = propensity_scores

# Perform matching based on propensity scores
def match(data, treatment_column, propensity_score_column):
    treated = data[data[treatment_column] == 1]
    control = data[data[treatment_column] == 0]
    
    matched_control = control.sample(n=len(treated), weights=1 / (1 - control[propensity_score_column]))
    
    matched_data = pd.concat([treated, matched_control])
    
    return matched_data

matched_data = match(lalonde, treatment, 'propensity_score')

# The matched_data dataframe now contains the matched data
print(matched_data)

from sklearn.linear_model import LogisticRegression

# Step 1: Calculate propensity scores
# Assuming you have your features for the logistic regression model in `X` and the 'Group' (treatment) in `y`
X = mydata[["Age", "Sex"]]  # Adjust the feature columns as needed
y = mydata["Group"]

# Fit a logistic regression model to calculate propensity scores
propensity_model = LogisticRegression()
propensity_model.fit(X, y)
propensity_scores = propensity_model.predict_proba(X)[:, 1]  # Use predicted probabilities of being in the treatment group

# Step 2: Calculate distances for matched pairs
# Assuming you have already calculated the `distance_matrix` as in your code

# Step 3: Add distance and propensity scores to the `matched_data` DataFrame
matched_data["Distance"] = distance_matrix[row_indices, col_indices]
matched_data["Propensity_Score"] = propensity_scores[treatment_group.index[row_indices]]

# Display the resulting matched_data DataFrame with added columns
print(matched_data)


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

