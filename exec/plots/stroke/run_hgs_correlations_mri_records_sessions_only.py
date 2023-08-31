import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import save_spearman_correlation_results
from hgsprediction.load_results import load_hgs_predicted_results, load_hgs_predicted_results_mri_records_sessions_only
from hgsprediction.load_results import load_spearman_correlation_results, load_spearman_correlation_results_mri_records_sessions_only
from hgsprediction.save_plot.save_correlations_plot.stroke_save_correlations_plot import save_correlations_plot, save_correlations_plot_mri_records_sessions_only
from hgsprediction.plots.plot_correlations import plot_hgs_correlations

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
model_name = sys.argv[7]
y = sys.argv[8]
x = sys.argv[9]
gender = sys.argv[10]

if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"

###############################################################################
df = load_hgs_predicted_results_mri_records_sessions_only(population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
)

df_corr, df_pvalue = load_spearman_correlation_results_mri_records_sessions_only (population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
)
print("===== Done! =====")
embed(globals(), locals())

###############################################################################
file_path = save_correlations_plot_mri_records_sessions_only(
    x,
    y,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
)

plot = plot_hgs_correlations(df,
                        x,
                        y,
                        stroke_cohort,
                        feature_type,
                        target,
                        gender)
plot.show()
plot.savefig(file_path)
plot.close()
print("===== Done! =====")
embed(globals(), locals())