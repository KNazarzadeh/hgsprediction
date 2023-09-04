import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def save_correlations_plot(
    plot_type,
    x,
    y,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
):
    # Assuming that you have already trained and instantiated the model as `model`
    folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "project_hgsprediction",  
            "plots",
            f"{population}",
            f"{mri_status}",
            f"{feature_type}",
            f"{target}",
            f"{model_name}",
            "hgs_correlations_plots",
            f"{plot_type}",
            f"{y}_vs_{x}",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_{y}_vs_{x}.png")
        
    return file_path

##############################################################################
