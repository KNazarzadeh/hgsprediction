import pandas as pd
import numpy as np
import sys

from hgsprediction.load_data.healthy import load_multi_samplesize_training_data

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
data_set = sys.argv[6]
###############################################################################
session = "0"
###############################################################################
samplesize_list = ["10_percent", "20_percent", "40_percent", "60_percent", "80_percent", "100_percent"]

for samplesize in samplesize_list:
    samplesize = f"{samplesize}_percent"
    df_female = load_multi_samplesize_training_data(
        population,
        mri_status,
        confound_status,
        "female",
        feature_type,
        target,
        session,
        data_set,    
        samplesize,
        )

    df_male = load_multi_samplesize_training_data(
        population,
        mri_status,
        confound_status,
        "male",
        feature_type,
        target,
        session,
        data_set,    
        samplesize,
        )

    #----------------------------------------------------------------------#
    print("Samplesize =", samplesize)
    
    whole_data_length = len(df_female)+len(df_male)
    print("\n Number of whole holdout test data N =", whole_data_length)


    print("\n Number of Female holdout test data N =", len(df_female))
    print("'%' of Female holdout test data N = %", "{:.2f}".format(len(df_female)*100/whole_data_length))

    print("\n Number of Male holdout test data N =", len(df_male))
    print("'%' of Male holdout test data N = %", "{:.2f}".format(len(df_male)*100/whole_data_length))

    print("=================================================")

print("===== Done! =====")
embed(globals(), locals())