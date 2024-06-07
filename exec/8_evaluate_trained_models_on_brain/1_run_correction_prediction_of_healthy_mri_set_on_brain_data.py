#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from hgsprediction.load_results.healthy import load_corrected_prediction_results
from hgsprediction.load_data.brain_correlate import load_original_brain_data
#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
data_set =sys.argv[10]
brain_data_type = sys.argv[11]
tiv_status = sys.argv[12]
schaefer = sys.argv[13]
stats_correlation_type = sys.argv[14]

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

df_female = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)

df_male = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)
###############################################################################

df = pd.concat([df_female, df_male], axis=0)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
df_brain = load_original_brain_data(brain_data_type, schaefer)
###############################################################################
tiv_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    f"TIV",
)

df_tiv = pd.read_csv(f"{tiv_path}/cat_rois_Schaefer2018_600Parcels_17Networks_order.csv", sep=',', index_col=0)

tiv = df_tiv[df_tiv['Session']=='ses-2']['TIV']

merged_gmv_tiv = pd.merge(brain_df, tiv , left_index=True, right_index=True, how='inner')

brain_regions = brain_df.columns
# Initialize a DataFrame to store residuals
residuals_df1 = pd.DataFrame(index=merged_gmv_tiv.index, columns=brain_regions)
# Loop through each region
for region in brain_regions:
    # Extract TIV values
    X = merged_gmv_tiv.loc[:, 'TIV'].values.reshape(-1, 1)
    # Extract the region's values
    y = merged_gmv_tiv.loc[:, region].values.reshape(-1, 1)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict using the model
    y_pred = model.predict(X)
    
    # Calculate residuals
    residuals = y - y_pred
    # Store residuals in the DataFrame
    residuals_df1.loc[:, region] = residuals

residuals_df1.index = residuals_df1.index.str.replace("sub-", "")
residuals_df1.index = residuals_df1.index.map(int)
##############################################################################
# Remove true HGS variance from regions:
true_hgs = df.loc[:, f'{target}']

merged_gmv_true = pd.merge(residuals_df1, true_hgs , left_index=True, right_index=True, how='inner')

# Initialize a DataFrame to store residuals
residuals_df2 = pd.DataFrame(index=merged_gmv_true.index, columns=brain_regions)

# Loop through each region
for region in brain_regions:
    # Extract TIV values
    X = merged_gmv_true.loc[:, f'{target}'].values.reshape(-1, 1)
    # Reshape X to have two columns
    # X = X.reshape(-1, 2)
    # Extract the region's values
    y = merged_gmv_true.loc[:, region].values.reshape(-1, 1)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict using the model
    y_pred = model.predict(X)
    
    # Calculate residuals
    residuals = y - y_pred
    # Store residuals in the DataFrame
    residuals_df2.loc[:, region] = residuals

##############################################################################
merged_df = pd.merge(residuals_df2, df, left_index=True, right_index=True, how='inner')

merged_df_female = merged_df[merged_df['gender']==0]
merged_df_male = merged_df[merged_df['gender']==1]



print("===== Done! =====")
embed(globals(), locals())
