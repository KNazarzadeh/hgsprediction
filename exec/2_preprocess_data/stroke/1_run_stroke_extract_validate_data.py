
import pandas as pd
import sys

from hgsprediction.load_data import stroke_load_data
from hgsprediction.data_preprocessing import stroke_data_preprocessor
from hgsprediction.save_data import stroke_save_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

df_original = stroke_load_data.load_original_data(population=population, mri_status=mri_status)

###############################################################################

data_processor = stroke_data_preprocessor.StrokeMainDataPreprocessor(df_original)
df = data_processor.remove_missing_stroke_dates(df_original)
df = data_processor.remove_missing_hgs(df)

###############################################################################
df = data_processor.define_stroke_type(df)
df = data_processor.define_followup_days(df)
###############################################################################
df_preprocessed = data_processor.preprocess_stroke_df(df)
# Calculate and Add dominant and nondominant hgs to data
df_preprocessed = data_processor.calculate_dominant_nondominant_hgs(df_preprocessed)
print("===== Done! =====")
embed(globals(), locals())
# Remove all columns with all NaN values
df_preprocessed = data_processor.remove_nan_columns(df_preprocessed)

# stroke_save_data.save_main_preprocessed_data(df_preprocessed, population, mri_status, stroke_cohort="all-stroke-subjects")

###############################################################################
for stroke_cohort in ["pre-stroke", "post-stroke"]:
    for visit_session in range(1, 4):
        if mri_status == "mri":
            if visit_session == 1:
                session_column = f"1st_{stroke_cohort}_session"
                df_extracted = data_processor.extract_data(df_preprocessed, session_column)
                df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            elif visit_session == 2:
                session_column = f"2nd_{stroke_cohort}_session"
                df_extracted = data_processor.extract_data(df_preprocessed, session_column)
                df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            elif visit_session == 3:
                session_column = f"3rd_{stroke_cohort}_session"
                df_extracted = data_processor.extract_data(df_preprocessed, session_column)
                df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            # stroke_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, stroke_cohort)
            # stroke_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, stroke_cohort)
        elif mri_status == "nonmri":
            if visit_session == 1:
                session_column = f"1st_{stroke_cohort}_session"
                df_extracted = data_processor.extract_data(df_preprocessed, session_column)
                df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            elif visit_session == 2:
                session_column = f"2nd_{stroke_cohort}_session"
                df_extracted = data_processor.extract_data(df_preprocessed, session_column)
                df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            # stroke_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, stroke_cohort)
            # stroke_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, stroke_cohort)
            print(stroke_cohort)
            print(visit_session, session_column)
            print(df_extracted)
            print(df_validated)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
stroke_cohort = "post-stroke"
df_post = data_processor.extract_post_stroke_df(df_preprocessed, mri_status)
# stroke_save_data.save_subgroups_only_preprocessed_data(df_post, population, mri_status, stroke_cohort="post-stroke")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{stroke_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{stroke_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{stroke_cohort}_session"
    df_extracted = data_processor.extract_data(df_post, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, session_column)
    # stroke_save_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, stroke_cohort="post-stroke")
    # stroke_save_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, stroke_cohort="post-stroke")
###############################################################################
stroke_cohort = "pre-stroke"
df_pre = data_processor.extract_pre_stroke_df(df_preprocessed, mri_status)
# stroke_save_data.save_subgroups_only_preprocessed_data(df_pre, population, mri_status, stroke_cohort="pre-stroke")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{stroke_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{stroke_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{stroke_cohort}_session"
    df_extracted = data_processor.extract_data(df_pre, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, session_column)
    stroke_save_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, stroke_cohort="pre-stroke")
    stroke_save_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, stroke_cohort="pre-stroke")
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
stroke_cohort = "longitudinal-stroke"
df_longitudinal = data_processor.extract_longitudinal_stroke_df(df_preprocessed, mri_status)
stroke_save_data.save_main_preprocessed_data(df_longitudinal, population, mri_status, stroke_cohort="longitudinal-stroke")
for visit_session in range(1, 2):
    if visit_session == 1:
        stroke_cohort = "pre-stroke"
        session_column = f"1st_{stroke_cohort}_session"
        df_extracted_pre = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_pre = data_processor.validate_handgrips(df_extracted_pre, session_column)
        stroke_cohort = "post-stroke"
        session_column = f"1st_{stroke_cohort}_session"
        df_extracted_post = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_post = data_processor.validate_handgrips(df_extracted_post, session_column)   
    
    # Assuming you have DataFrames called 'df_pre' and 'df_post'
    # Concatenate the DataFrames, but keep only one of the columns with the same name
    merged_df_extracted = pd.concat([df_extracted_pre, df_extracted_post], axis=1, join="inner")
    merged_df_validated = pd.concat([df_validated_pre, df_validated_post], axis=1, join="inner")
    # Select which columns to keep (for example, keep columns from df_pre)
    df_longitudinal_extracted = merged_df_extracted.loc[:, ~merged_df_extracted.columns.duplicated()]
    df_longitudinal_validated = merged_df_validated.loc[:, ~merged_df_validated.columns.duplicated()]
    stroke_cohort = "longitudinal-stroke"
    session_column = f"1st_{stroke_cohort}_session"
    stroke_save_data.save_primary_extracted_data(df_longitudinal_extracted, population, mri_status, session_column, stroke_cohort="longitudinal-stroke")
    stroke_save_data.save_validated_hgs_data(df_longitudinal_validated, population, mri_status, session_column, stroke_cohort="longitudinal-stroke")

print("===== Done! =====")
embed(globals(), locals())

