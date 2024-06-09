import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def remove_true_hgs_variance_from_regions(
    df,
    target,
    brain_regions,
    
):
    
    # Initialize a DataFrame to store residuals
    df_residuals = pd.DataFrame(index=df.index, columns=brain_regions)

    # Loop through each region
    for region in brain_regions:
        # Extract TIV values
        X = df.loc[:, f'{target}'].values.reshape(-1, 1)
        # Reshape X to have two columns
        # X = X.reshape(-1, 2)
        # Extract the region's values
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
        
    return df_residuals