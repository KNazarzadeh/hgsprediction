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
session = sys.argv[3]
data_set = sys.argv[4]
###############################################################################
# load data # Extract data based on main features, extra features, target for each session and mri status:
df = load_healthy_data.load_original_nonmri_test_data(population, mri_status)

df_female = df[df['31-0.0'] == 0]
df_male = df[df['31-0.0'] == 1]
###############################################################################
whole_data_length = len(df_female)+len(df_male)
print("\n Number of whole Original Training data N=", whole_data_length)

print("\n Number of Female Original Training data N=", len(df_female))
print("'%' of Female Original Trainingt data N=%", "{:.2f}".format(len(df_female)*100/whole_data_length))

print("\n Number of Male Original Training data N=", len(df_male))
print("'%' of Male Original Training data N=%", "{:.2f}".format(len(df_male)*100/whole_data_length))

female_describe = df_female.describe().apply(lambda x: round(x, 2))
print("\n Female Describe=\n", female_describe)

male_describe = df_male.describe().apply(lambda x: round(x, 2))
print("\n Male Describe=\n", male_describe)

print("===== Done! =====")
embed(globals(), locals())