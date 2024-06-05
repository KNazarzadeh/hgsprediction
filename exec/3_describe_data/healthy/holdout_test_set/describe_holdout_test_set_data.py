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
df_female = load_healthy_data.load_extracted_data_by_feature_and_target(
    population,
    mri_status,
    feature_type,
    target,
    session,
    "female",
    data_set,
)
df_male = load_healthy_data.load_extracted_data_by_feature_and_target(
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
print("\n Number of whole holdout test data N=", whole_data_length)

print("\n Number of Female holdout test data N=", len(df_female))
print("'%' of Female holdout test data N=%", "{:.2f}".format(len(df_female)*100/whole_data_length))

print("\n Number of Male holdout test data N=", len(df_male))
print("'%' of Male holdout test data N=%", "{:.2f}".format(len(df_male)*100/whole_data_length))

print("\n Female Describe=\n", df_female.describe().apply(lambda x: round(x, 2)))

print("\n Male Describe=\n", df_male.describe().apply(lambda x: round(x, 2)))

female_right_handed = len(df_female[df_female['handedness-0.0']==1.0])
male_right_handed = len(df_male[df_male['handedness-0.0']==1.0])

print("'%' of Female Right dominant hand =", "{:.2f}".format(female_right_handed*100/len(df_female)))
print("'%' of Male Right dominant hand =", "{:.2f}".format(male_right_handed*100/len(df_male)))

print("===== Done! =====")
embed(globals(), locals())