

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def predict_hgs(df, X, y, best_model_trained):
    
    # y_True
    df["hgs_actual"] = df[y]
    # y_predicted
    df["hgs_predicted"] = best_model_trained.predict(df[X])
    # error: (actual-predicted)
    df["hgs_(actual-predicted)"] =  df["hgs_actual"] - df["hgs_predicted"]
    
    return df