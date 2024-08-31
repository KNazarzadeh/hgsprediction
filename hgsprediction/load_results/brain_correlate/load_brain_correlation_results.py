import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
def load_hgs_correlation_with_brain_regions_results(
    brain_data_type,
    schaefer,
    session,
    gender,
    corr_target,
):
    

    folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "project_hgsprediction",  
            "results_hgsprediction",
            "brain_correlation_results",
            f"{brain_data_type.upper()}_subcorticals_cerebellum",
            f"schaefer{schaefer}",
            "hgs_correlation_with_brain_regions_results",
            f"{session}_session_ukb",
            f"{corr_target}", 
        )
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_correlation_results.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)

    return df
###############################################################################
