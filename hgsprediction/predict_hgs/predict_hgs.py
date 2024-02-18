

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def predict_hgs(df, X, y, best_model_trained, session, target):
    
    # y_True
    df.loc[:, f"{target}_true"] = df.loc[:, y].copy()
    # y_predicted
    df.loc[:, f"{target}_predicted"] = best_model_trained.predict(df.loc[:, X]).copy()
    # error: (true-predicted)
    df.loc[:, f"{target}_(true-predicted)"] =  df.loc[:, f"{target}_true"].copy() - df.loc[:, f"{target}_predicted"].copy()
    
    return df