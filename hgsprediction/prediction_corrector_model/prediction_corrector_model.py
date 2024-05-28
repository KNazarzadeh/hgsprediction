import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from hgsprediction.load_results.load_trained_model_results import load_prediction_hgs_on_validation_set

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
def prediction_corrector_model(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
):
    
    n_repeats = 1
    n_folds = 10
    
    df = load_prediction_hgs_on_validation_set(
        population,
        mri_status,
        model_name,
        feature_type,
        target,
        gender,
        confound_status,
        n_repeats,
        n_folds,
        )


    model = LinearRegression()
    # Beheshti Method:
    X = df.loc[:, f"{target}"].values.reshape(-1, 1)
    y = df.loc[:, f"{target}_delta(true-predicted)"].values

    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_   
    
    return slope, intercept