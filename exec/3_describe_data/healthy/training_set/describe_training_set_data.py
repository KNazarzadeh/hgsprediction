import pandas as pd
import numpy as np
import sys

from hgsprediction.load_data.healthy import load_healthy_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
session = sys.argv[5]
data_set = sys.argv[6]
###############################################################################
# load data # Extract data based on main features, extra features, target for each session and mri status:
df_male = load_healthy_data.load_extracted_data_by_feature_and_target(
    population,
    mri_status,
    feature_type,
    target,
    session,
    "female",
    data_set,
)
df_female = load_healthy_data.load_extracted_data_by_feature_and_target(
    population,
    mri_status,
    feature_type,
    target,
    session,
    "male",
    data_set,
)
###############################################################################
whole_data_length = len(df_female)+len(df_male)
print("Number of whole holdout test data N=", whole_data_length)

print("Number of Female holdout test data N=", len(df_female))
print("'%' of Female holdout test data N=%", "{:.2f}".format(len(df_female)*100/whole_data_length))

print("Number of Male holdout test data N=", len(df_male))
print("'%' of Male holdout test data N=%", "{:.2f}".format(len(df_male)*100/whole_data_length))



print("===== Done! =====")
embed(globals(), locals())