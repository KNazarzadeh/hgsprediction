

import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def predict_hgs(df, X, y, best_model_trained, target):
    
    # y_True --> df.loc[:, f"{target}"]
    # y_predicted
    df.loc[:, f"{target}_predicted"] = best_model_trained.predict(df.loc[:, X])
    # error: (true-predicted)
    df.loc[:, f"{target}_delta(true-predicted)"] =  df.loc[:, f"{target}"] - df.loc[:, f"{target}_predicted"]
    # df.loc[:, f"{target}_delta(predicted-true)"] =  df.loc[:, f"{target}_predicted"] - df.loc[:, f"{target}"]

    return df