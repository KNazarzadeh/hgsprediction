import os
import pandas as pd
import numpy as np
import sys
from hgsprediction.load_results import healthy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import datatable as dt
from hgsprediction.predict_hgs import calculate_brain_hgs                    
from hgsprediction.predict_hgs import calculate_t_valuesGMV_HGS

from sklearn.metrics import r2_score 
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

from hgsprediction.save_results.brain_save_correlates_results import save_brain_hgs_correlation_results, save_brain_overlap_data_results
from nilearn import datasets
import nibabel as nib

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
brain_data_type = sys.argv[10]
schaefer = sys.argv[11]
stats_correlation_type = sys.argv[12]
###############################################################################
jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    f"{brain_data_type.upper()}",
)

schaefer_file = os.path.join(jay_path, f"{brain_data_type.upper()}_Schaefer{schaefer}x7_Mean.jay")

dt_schaefer = dt.fread(schaefer_file)
brain_df = dt_schaefer.to_pandas()
brain_df.set_index('SubjectID', inplace=True)

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
residuals_df = pd.DataFrame(index=merged_gmv_tiv.index, columns=brain_regions)
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
    residuals_df.loc[:, region] = residuals

residuals_df.index = residuals_df.index.str.replace("sub-", "")
residuals_df.index = residuals_df.index.map(int)
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
visual_net_list = []
somatomotor_net_list = []
dorsal_net_list = []
salience_ventral_net_list = []
limbic_net_list = []
control_net_list = []
default_net_list = []

def extract_substring_between_underscores(input_string):
    # Find the index of the first underscore
    first_underscore_index = input_string.find("_")
    
    # Find the index of the second underscore
    second_underscore_index = input_string.find("_", first_underscore_index + 1)
    
    # If either underscore is not found, return an empty string
    if first_underscore_index == -1 or second_underscore_index == -1:
        return ""
    
    # Extract the substring between the first two underscores
    substring_between_underscores = input_string[first_underscore_index + 1:second_underscore_index]
    
    return substring_between_underscores

for region in brain_regions:
    result = extract_substring_between_underscores(region)
    if result == "Vis":
       visual_net_list.append(region)
    elif result == "SomMot":
           somatomotor_net_list.append(region)
    elif result == "DorsAttn":
           dorsal_net_list.append(region)
    elif result == "SalVentAttn":
           salience_ventral_net_list.append(region)
    elif result == "Cont":
           control_net_list.append(region)      
    elif result == "Limbic":
           limbic_net_list.append(region)
    elif result == "Default":
           default_net_list.append(region)

visual_net = residuals_df[visual_net_list]
somatomotor_net = residuals_df[somatomotor_net_list]
dorsal_net = residuals_df[dorsal_net_list]
salience_ventral_net = residuals_df[salience_ventral_net_list]
control_net = residuals_df[control_net_list]
limbic_net = residuals_df[limbic_net_list]
default_net = residuals_df[default_net_list]

# Calculate row-wise average
# Add the row-wise average as a new column
visual_net['visual_net_average'] = visual_net.mean(axis=1)
somatomotor_net['somatomotor_net_average'] = somatomotor_net.mean(axis=1)
dorsal_net['dorsal_net_average'] = dorsal_net.mean(axis=1)
salience_ventral_net['salience_ventral_net_average'] = salience_ventral_net.mean(axis=1)
control_net['control_net_average'] = control_net.mean(axis=1)
limbic_net['limbic_net_average'] = limbic_net.mean(axis=1)
default_net['default_net_average'] = default_net.mean(axis=1)

##############################################################################
# load data
df = healthy.load_hgs_predicted_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
    confound_status,
    n_repeats,
    n_folds,    
)

merged_df = pd.merge(residuals_df, df, left_index=True, right_index=True, how='inner')

merged_df_female = merged_df[merged_df['gender']==0]
merged_df_male = merged_df[merged_df['gender']==1]
