

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def predict_hgs(df, X, y, best_model_trained):
    
    y_true = df[y]
    y_pred = best_model_trained.predict(df[X])
    df.loc[:, "hgs_actual"] = y_true
    df.loc[:, "hgs_predicted"] = y_pred
    df.loc[:, "hgs_(actual-predicted)"] = y_true - y_pred
    df.loc[:, "years"] = df.loc[:, "days"]/365
    
    return df