import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from hgsprediction.load_results.healthy import load_trained_model_results

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
def prediction_corrector_model(
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
):
    population = "healthy"
    mri_status = "nonmri"
    confound_status = "0"
    n_repeats = 1
    n_folds = 10
    session= 0
    data_set="training_set"
    
    df = load_trained_model_results.load_prediction_hgs_on_validation_set(
        population,
        mri_status,
        confound_status,
        gender,
        feature_type,
        target,
        model_name,
        n_repeats,
        n_folds,
        session,
        data_set,
        )


    model = LinearRegression()

    # Beheshti Method:
    X = df.loc[:, f"{target}"].values.reshape(-1, 1)
    y = df.loc[:, f"{target}_delta(true-predicted)"].values

    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_   
    
    return slope, intercept