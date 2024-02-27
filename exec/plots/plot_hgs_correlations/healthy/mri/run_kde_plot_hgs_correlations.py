import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import save_spearman_correlation_results
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
target = sys.argv[4]
model_name = sys.argv[5]
y = sys.argv[6]
x = sys.argv[7]
session = sys.argv[8]
print("===== Done! =====")
embed(globals(), locals())

###############################################################################
df = load_hgs_predicted_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
)

df_corr, df_pvalue = load_spearman_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
)

###############################################################################
df = df.rename(columns={f"1st_scan_{target}_predicted":"hgs_predicted", f"1st_scan_{target}_actual":"hgs_actual"})
df_corr = df_corr.rename(columns={f"1st_scan_{target}_predicted":"hgs_predicted", f"1st_scan_{target}_actual":"hgs_actual"})
df_corr = df_corr.rename(index={f"1st_scan_{target}_predicted":"hgs_predicted", f"1st_scan_{target}_actual":"hgs_actual"})


healthy_plot_hgs_correlations.plot_hgs_correlations_kde_plot(
    df, 
    x,
    y,
    population,
    mri_status,
    model_name,
    feature_type,
    target,)

print("===== Done! =====")
embed(globals(), locals())