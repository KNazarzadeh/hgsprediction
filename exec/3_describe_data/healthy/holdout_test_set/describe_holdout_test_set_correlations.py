
import sys
import pandas as pd
import numpy as np

from hgsprediction.load_results.healthy.load_prediction_correlation_results import load_prediction_correlation_results

#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
data_set = sys.argv[10]
correlation_type = sys.argv[11]

###############################################################################
df_female_correlation_values, df_female_r2_values = load_prediction_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,    
    correlation_type,
    data_set,
)

df_male_correlation_values, df_male_r2_values = load_prediction_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,    
    correlation_type,
    data_set,
)
###############################################################################
# Apply formatting to each element in the DataFrame
formatted_df_male_correlation_values = df_male_correlation_values.applymap(lambda x: "{:.2f}".format(x))
print("Male Correlations=\n", formatted_df_male_correlation_values)

formatted_df_male_r2_values = df_male_r2_values.applymap(lambda x: "{:.2f}".format(x))
print("Male R2 =\n", formatted_df_male_r2_values)


# Apply formatting to each element in the DataFrame
formatted_df_female_correlation_values = df_female_correlation_values.applymap(lambda x: "{:.2f}".format(x))
print("Female Correlations=\n", formatted_df_female_correlation_values)

formatted_df_female_r2_values = df_female_r2_values.applymap(lambda x: "{:.2f}".format(x))
print("Female R2 =\n", formatted_df_female_r2_values)





print("===== Done! =====")
embed(globals(), locals())