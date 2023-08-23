import sys
import os
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.compute_features import StrokeFeaturesComputing
from hgsprediction.save_data import stroke_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]

if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"

df = stroke_load_data.load_validated_hgs_data(population, mri_status, session_column)


data_processor = StrokeFeaturesComputing(df, mri_status, feature_type, stroke_cohort, visit_session)

# Call all functions inside the class
# FEATURE ENGINEERING
df = data_processor.calculate_bmi(df)
df = data_processor.calculate_height(df)
df = data_processor.calculate_waist_to_hip_ratio(df)
df = data_processor.calculate_age(df)
df = data_processor.calculate_days(df)

print("===== Done! =====")
embed(globals(), locals())

stroke_save_data.save_computed_features_data(df, population, mri_status, session_column)
