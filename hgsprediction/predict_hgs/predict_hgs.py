

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def predict_hgs(df, X, y, best_model_trained, target):
    
    # y_True
    df.loc[:, f"{target}_actual"] = df.loc[:, y]
    # y_predicted
    df.loc[:, f"{target}_predicted"] = best_model_trained.predict(df.loc[:, X])
    # error: (actual-predicted)
    df.loc[:, f"{target}_(actual-predicted)"] =  df.loc[:, f"{target}_actual"] - df.loc[:, f"{target}_predicted"]
    
    return df