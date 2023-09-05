import sys
import os
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.save_data import stroke_save_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]

###############################################################################
df_main_preprocessed_longitudinal = stroke_load_data.load_main_preprocessed_data(population, mri_status, stroke_group="only_longitudinal-stroke")
###############################################################################
stroke_cohort = "pre-stroke"
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"

df_pre = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, gender="both_gender")
###############################################################################
stroke_cohort = "post-stroke"
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
df_post = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, gender="both_gender")
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
merged_df = pd.concat([df_pre, df_post], axis=1, join="inner")
df = merged_df[merged_df.index.isin(df_main_preprocessed_longitudinal.index)]
###############################################################################
if len(df) > 0:
    df_female = df[df["31-0.0"]==0.0]
    df_male = df[df["31-0.0"]==1.0]
    
    stroke_cohort = "pre-post-stroke"
    if visit_session == "1":
        session_column = f"1st_{stroke_cohort}_session"
    elif visit_session == "2":
        session_column = f"2nd_{stroke_cohort}_session"
    elif visit_session == "3":
        session_column = f"3rd_{stroke_cohort}_session"

print("===== Done! =====")
embed(globals(), locals())

stroke_save_data.save_preprocessed_longitudinal_data(df, population, mri_status, session_column, "both_gender")
stroke_save_data.save_preprocessed_longitudinal_data(df_female, population, mri_status, session_column, "female")
stroke_save_data.save_preprocessed_longitudinal_data(df_male, population, mri_status, session_column, "male")

print("===== Done! =====")
embed(globals(), locals())