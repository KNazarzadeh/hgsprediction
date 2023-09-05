import sys
import os
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.save_data import stroke_save_data
from hgsprediction.compute_target import stroke_compute_target

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]
# target = sys.argv[5]
# gender = sys.argv[6]

if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"
    
df = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, stroke_cohort, "both_gender")

for target in ["hgs_L+R", "hgs_left", "hgs_right", "hgs_LI", "hgs_L-R"]:
    df = stroke_compute_target.compute_target(df, session_column, target)

    df_female = df[df["31-0.0"]==0.0]
    df_male = df[df["31-0.0"]==1.0]

    stroke_save_data.save_preprocessed_data(df, population, mri_status, session_column, stroke_cohort, "both_gender")
    stroke_save_data.save_preprocessed_data(df_female, population, mri_status, session_column, stroke_cohort, "female")
    stroke_save_data.save_preprocessed_data(df_male, population, mri_status, session_column, stroke_cohort, "male")

print("===== Done! =====")
embed(globals(), locals())