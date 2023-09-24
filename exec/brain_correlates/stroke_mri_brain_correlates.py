import os
import pandas as pd
import numpy as np
import sys
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.load_imaging_data import load_imaging_data
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# from hgsprediction.plots import create_regplot

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

###############################################################################

female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "female",
                                feature_type,
                                target,
                                "linear_svm",
                                10,
                                5,
                            )

print(female_best_model_trained)

male_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "male",
                                feature_type,
                                target,
                                "linear_svm",
                                10,
                                5,
                            )
print(male_best_model_trained)

##############################################################################
# load data
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
df_longitudinal = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, stroke_cohort)

features = define_features(feature_type)
X = features
y = target

print("===== Done! =====")
embed(globals(), locals())

###############################################################################################################################################################
