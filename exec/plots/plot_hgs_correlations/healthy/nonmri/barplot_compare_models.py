import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results.disorder import save_spearman_correlation_results
from hgsprediction.load_results.healthy import load_hgs_predicted_results
from hgsprediction.load_results.healthy import load_spearman_correlation_results
from hgsprediction.save_plot.save_correlations_plot import healthy_save_correlations_plot
from hgsprediction.plots.plot_correlations import healthy_plot_hgs_correlations

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
session = sys.argv[5]
###############################################################################
df_combine = pd.DataFrame()
for tar in ["hgs_left", "hgs_right", "hgs_L+R"]:
    df = load_hgs_predicted_results(
        population,
        mri_status,
        model_name,
        feature_type,
        tar,
        "both_gender",
        session,
    )
    df.loc[:, 'target'] = tar
    df = df.rename(columns={f"1st_scan_{tar}_predicted":"hgs_predicted", f"1st_scan_{tar}_actual":"hgs_actual", f"1st_scan_{tar}_(actual-predicted)":"actual-predicted"})

    df_combine = pd.concat([df_combine, df[['1st_scan_age', 'gender', 'hgs_actual', 'hgs_predicted', 'actual-predicted', 'target']]])

df_combine['gender'] = df_combine['gender'].replace({0: 'Female', 1: 'Male'})
print("===== Done! =====")
embed(globals(), locals())