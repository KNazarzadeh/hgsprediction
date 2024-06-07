import os
import pandas as pd
import numpy as np


def load_tiv_data():

    tiv_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "brain_imaging_data",
        "TIV",
    )

    df_tiv = pd.read_csv(f"{tiv_path}/cat_rois_Schaefer2018_600Parcels_17Networks_order.csv", sep=',', index_col=0)

    return df_tiv