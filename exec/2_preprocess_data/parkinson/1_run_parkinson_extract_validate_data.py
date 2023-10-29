import pandas as pd
import sys

from hgsprediction.load_data import parkinson_load_data
from hgsprediction.data_preprocessing import parkinson_data_preprocessor
from hgsprediction.save_data import parkinson_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

pd.options.mode.chained_assignment = None  # 'None' suppresses the warning

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

df_original = parkinson_load_data.load_original_data(population, mri_status)

###############################################################################
data_processor = parkinson_data_preprocessor.ParkinsonMainDataPreprocessor(df_original)
df = data_processor.remove_missing_parkinson_dates(df_original)
df = data_processor.remove_missing_hgs(df)

###############################################################################
df = data_processor.define_followup_days(df)

###############################################################################
###############################################################################
df_preprocessed = data_processor.preprocess_parkinson_df(df)

# Calculate and Add dominant and nondominant hgs to data
df_preprocessed = data_processor.calculate_dominant_nondominant_hgs(df_preprocessed)
# Remove all columns with all NaN values
df_preprocessed = data_processor.remove_nan_columns(df_preprocessed)

parkinson_save_data.save_main_preprocessed_data(df_preprocessed, population, mri_status, parkinson_cohort="all-parkinson-subjects")

###############################################################################
for parkinson_cohort in ["pre-parkinson", "post-parkinson"]:
    for visit_session in range(1, 3):
        if mri_status == "mri":
            if visit_session == 1:
                session_column = f"1st_{parkinson_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{parkinson_cohort}_session"
            elif visit_session == 3:
                session_column = f"3rd_{parkinson_cohort}_session"
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            print(df_extracted)
            print(df_validated)
            parkinson_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, parkinson_cohort)
            parkinson_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, parkinson_cohort)
        elif mri_status == "nonmri":
            if visit_session == 1:
                session_column = f"1st_{parkinson_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{parkinson_cohort}_session"
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            parkinson_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, parkinson_cohort)
            parkinson_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, parkinson_cohort)
            print(df_extracted)
            print(df_validated)

###############################################################################
parkinson_cohort = "post-parkinson"
df_post = data_processor.extract_post_parkinson_df(df_preprocessed, mri_status)
parkinson_save_data.save_subgroups_only_preprocessed_data(df_post, population, mri_status, parkinson_cohort="post-parkinson")
if mri_status == "mri":
    visit_range = range(1, 3)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{parkinson_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{parkinson_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{parkinson_cohort}_session"
    df_extracted = data_processor.extract_data(df_post, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, session_column)
    print(df_extracted)
    print(df_validated)

###############################################################################
parkinson_cohort = "pre-parkinson"
df_pre = data_processor.extract_pre_parkinson_df(df_preprocessed, mri_status)
parkinson_save_data.save_subgroups_only_preprocessed_data(df_pre, population, mri_status, parkinson_cohort="pre-parkinson")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{parkinson_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{parkinson_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{parkinson_cohort}_session"
    df_extracted = data_processor.extract_data(df_pre, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, session_column)
    parkinson_save_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, parkinson_cohort="pre-parkinson")
    parkinson_save_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, parkinson_cohort="pre-parkinson")
    print(df_extracted)
    print(df_validated)

###############################################################################
parkinson_cohort = "longitudinal-parkinson"
df_longitudinal = data_processor.extract_longitudinal_parkinson_df(df_preprocessed, mri_status)
parkinson_save_data.save_main_preprocessed_data(df_longitudinal, population, mri_status, parkinson_cohort="longitudinal-parkinson")
for visit_session in range(1, 2):
    if visit_session == 1:
        parkinson_cohort = "pre-parkinson"
        session_column = f"1st_{parkinson_cohort}_session"
        df_extracted_pre = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_pre = data_processor.validate_handgrips(df_extracted_pre, session_column)
        parkinson_cohort = "post-parkinson"
        session_column = f"1st_{parkinson_cohort}_session"
        df_extracted_post = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_post = data_processor.validate_handgrips(df_extracted_post, session_column)   
    
    # Assuming you have DataFrames called 'df_pre' and 'df_post'
    # Concatenate the DataFrames, but keep only one of the columns with the same name
    merged_df_extracted = pd.concat([df_extracted_pre, df_extracted_post], axis=1, join="inner")
    merged_df_validated = pd.concat([df_validated_pre, df_validated_post], axis=1, join="inner")
    # Select which columns to keep (for example, keep columns from df_pre)
    df_longitudinal_extracted = merged_df_extracted.loc[:, ~merged_df_extracted.columns.duplicated()]
    df_longitudinal_validated = merged_df_validated.loc[:, ~merged_df_validated.columns.duplicated()]
    parkinson_cohort = "longitudinal-parkinson"
    session_column = f"1st_{parkinson_cohort}_session"
    parkinson_save_data.save_primary_extracted_data(df_longitudinal_extracted, population, mri_status, session_column, parkinson_cohort="longitudinal-parkinson")
    parkinson_save_data.save_validated_hgs_data(df_longitudinal_validated, population, mri_status, session_column, parkinson_cohort="longitudinal-parkinson")
    
print("===== Done! =====")
embed(globals(), locals())
