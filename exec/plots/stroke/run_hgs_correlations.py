import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import save_spearman_results

from hgsprediction.save_plot import stroke_save_correlations_plot
from hgsprediction.plots import plot_correlations

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
x = sys.argv[8]
y = sys.argv[9]
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
df = 
file_path = stroke_save_correlations_plot(
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

plot = plot_correlations(df,
                        x=x,
                        y=y,
                        stroke_cohort,
                        feture_type,
                        target,
                        gender)
plot.show()
plot.savefig(file_path)
plot.close()
