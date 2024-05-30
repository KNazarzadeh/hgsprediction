
import sys
import pandas as pd
import numpy as np

from hgsprediction.load_results.healthy.load_multi_samples_trained_models_results import load_scores_trained

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Inputs : Required inputs
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
confound_status = sys.argv[5]
n_repeats = sys.argv[6]
n_folds = sys.argv[7]
data_set = sys.argv[8]
score_type = sys.argv[9]
###############################################################################
session = "0"
###############################################################################
if score_type == "r_score":
    test_score = "test_pearson_corr"
elif score_type == "r2_score":
    test_score = "test_r2"
###############################################################################
print("\n Score Type =", score_type)
###############################################################################
samplesize_list = ["10_percent", "20_percent", "40_percent", "60_percent", "80_percent", "100_percent"]

for model_name in ["linear_svm", "random_forest"]:
    for samplesize in samplesize_list:
        df_female = load_scores_trained(
            population,
            mri_status,
            confound_status,
            "female",
            feature_type,
            target,
            model_name,
            n_repeats,
            n_folds,
            session,
            data_set,
            samplesize,
            )
        df_male = load_scores_trained(
            population,
            mri_status,
            confound_status,
            "male",
            feature_type,
            target,
            model_name,
            n_repeats,
            n_folds,
            session,
            data_set,
            samplesize,
            )
        
        #----------------------------------------------------------------------#

        print("\n Samplesize =", samplesize)
        print("\n Model =", model_name)

        print(f"\n Female Median {score_type}: {model_name} {samplesize} = {np.median(df_female[test_score]):.3f}")
        
        print(f"\n Male Median {score_type}: {model_name} {samplesize} = {np.median(df_male[test_score]):.3f}")
        
        print("=================================================")