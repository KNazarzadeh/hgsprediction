import sys
import os
import pandas as pd
from hgsprediction.load_data import stroke_load_data
from hgsprediction.save_data import stroke_save_data
from hgsprediction.define_features import define_features


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
feature_type = sys.argv[4]
visit_session = sys.argv[5]

###############################################################################
df_main_preprocessed_longitudinal = stroke_load_data.load_main_preprocessed_data(population, mri_status, stroke_group="only_longitudinal-stroke")
features = define_features(feature_type)

###############################################################################
stroke_cohort = "pre-stroke"
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
df_pre = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, stroke_cohort, gender="both_gender")
###############################################################################
stroke_cohort = "post-stroke"
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
df_post = stroke_load_data.load_preprocessed_data(population, mri_status, session_column, stroke_cohort, gender="both_gender")

###############################################################################
# Assuming you have DataFrames called 'df_pre' and 'df_post'
# Concatenate the DataFrames, but keep only one of the columns with the same name
merged_df = pd.concat([df_pre, df_post], axis=1, join="inner")
# Select which columns to keep (for example, keep columns from df_pre)
df = merged_df.loc[:, ~merged_df.columns.duplicated()]
###############################################################################
features_columns = [col for col in df.columns if any(col.endswith(item) for item in features)]
extracted_df1 = df[features_columns].copy()
extracted_df = extracted_df1.dropna(subset=[col for col in df.columns if any(item in col for item in features)])

df = df[df.index.isin(extracted_df.index)]
print("===== Done! =====")
embed(globals(), locals())
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

stroke_save_data.save_preprocessed_longitudinal_data(df, population, mri_status, session_column, "both_gender")
stroke_save_data.save_preprocessed_longitudinal_data(df_female, population, mri_status, session_column, "female")
stroke_save_data.save_preprocessed_longitudinal_data(df_male, population, mri_status, session_column, "male")

print("===== Done! =====")
embed(globals(), locals())