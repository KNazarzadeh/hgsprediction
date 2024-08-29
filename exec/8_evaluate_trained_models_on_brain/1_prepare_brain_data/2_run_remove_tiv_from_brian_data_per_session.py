
import pandas as pd
import numpy as np
import sys

from sklearn.linear_model import LinearRegression
from hgsprediction.load_data.brain_correlate import load_original_brain_data
from hgsprediction.load_data.brain_correlate import load_tiv_data
from hgsprediction.save_data.brain_correlate.save_removed_tiv_from_brain_data import save_removed_tiv_from_brain_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
session = sys.argv[3]
brain_data_type = sys.argv[4]
schaefer = sys.argv[5]

###############################################################################
df_brain = load_original_brain_data(brain_data_type, schaefer)
###############################################################################
df_tiv_original = load_tiv_data()
##############################################################################
if session == '2':
    df_tiv = df_tiv_original[df_tiv_original['Session'] == f'ses-2']['TIV']
elif session == '3':
    df_tiv_original_ses_2 = df_tiv_original[df_tiv_original['Session'] == f'ses-2']['TIV']
    df_tiv_original_ses_3 = df_tiv_original[df_tiv_original['Session'] == f'ses-3']['TIV']

    df_tiv_ses_3 = df_tiv_original_ses_3[~df_tiv_original_ses_3.index.isin(df_tiv_original_ses_2.index)]
    df_tiv = df_tiv_ses_3
##############################################################################
def calculate_residuals(df, brain_regions):
    df_residuals = pd.DataFrame(index=df.index, columns=brain_regions)
    for region in brain_regions:
        X = df.loc[:, 'TIV'].values.reshape(-1, 1)
        y = df.loc[:, region].values.reshape(-1, 1)
        # Fit linear regression model
        model = LinearRegression()
        
        model.fit(X, y)
        # Predict using the model        
        y_pred = model.predict(X)
        # Calculate residuals
        residuals = y - y_pred
        # Store residuals in the DataFrame
        df_residuals.loc[:, region] = residuals

    df_residuals.index = df_residuals.index.str.replace("sub-", "")
    df_residuals.index = df_residuals.index.map(int)
    
    return df_residuals
##############################################################################
    
df_merged_gmv_tiv = pd.merge(df_brain, df_tiv, left_index=True, right_index=True, how='inner')

if len(df_merged_gmv_tiv) > 0:

    brain_regions = df_brain.columns

    # Calculate residuals for mri session (ses_2 or ses-3)
    df_residuals = calculate_residuals(df_merged_gmv_tiv, brain_regions)

    # Display the residuals DataFrames
    print(df_residuals)
    
    save_removed_tiv_from_brain_data(
        df_residuals,
        session,
        brain_data_type,
        schaefer,   
    )
else:
    print("There is no subject with brain data for this session")

print("===== Done! =====")
embed(globals(), locals())