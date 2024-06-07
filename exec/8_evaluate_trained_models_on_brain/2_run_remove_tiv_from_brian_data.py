
import pandas as pd
import numpy as np
import sys

from sklearn.linear_model import LinearRegression
from hgsprediction.load_data.brain_correlate import load_original_brain_data
from hgsprediction.load_data.brain_correlate import load_tiv_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
brain_data_type = sys.argv[1]
schaefer = sys.argv[2]

###############################################################################
df_brain = load_original_brain_data(brain_data_type, schaefer)
###############################################################################

df_tiv = load_tiv_data()
##############################################################################

df_tiv_ses_2 = df_tiv[df_tiv['Session']=='ses-2']['TIV']
df_tiv_ses_3 = df_tiv[df_tiv['Session']=='ses-3']['TIV']


df_merged_gmv_tiv_ses_2 = pd.merge(df_brain, df_tiv_ses_2 , left_index=True, right_index=True, how='inner')
df_merged_gmv_tiv_ses_3 = pd.merge(df_brain, df_tiv_ses_3 , left_index=True, right_index=True, how='inner')

brain_regions = df_brain.columns
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
# Calculate residuals for ses_2
df_residuals_ses_2 = calculate_residuals(df_merged_gmv_tiv_ses_2, brain_regions)

# Calculate residuals for ses_3
df_residuals_ses_3 = calculate_residuals(df_merged_gmv_tiv_ses_3, brain_regions)

# Display the residuals DataFrames
print(df_residuals_ses_2.head())
print(df_residuals_ses_3.head())

##############################################################################
save_brain_{brain_data_type}_without_tiv(
    brain_data_type,
    schaefer,
    
)

print("===== Done! =====")
embed(globals(), locals())