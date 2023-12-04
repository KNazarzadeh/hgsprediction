import pandas as pd
import sys

from hgsprediction.load_data import depression_load_data
from hgsprediction.data_preprocessing import depression_data_preprocessor
from hgsprediction.save_data import depression_save_data
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

pd.options.mode.chained_assignment = None  # 'None' suppresses the warning

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

df_original = depression_load_data.load_original_data(population, mri_status)

###############################################################################
data_processor = depression_data_preprocessor.DepressionMainDataPreprocessor(df_original)
df = data_processor.remove_missing_depression_dates(df_original)
df = data_processor.remove_missing_hgs(df)

###############################################################################
df = data_processor.define_followup_days(df)

###############################################################################
###############################################################################
df_preprocessed = data_processor.preprocess_depression_df(df)

# Calculate and Add dominant and nondominant hgs to data
df_preprocessed = data_processor.calculate_dominant_nondominant_hgs(df_preprocessed)
# Remove all columns with all NaN values
df_preprocessed = data_processor.remove_nan_columns(df_preprocessed)

depression_save_data.save_main_preprocessed_data(df_preprocessed, population, mri_status, depression_cohort="all-depression-subjects")
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
for depression_cohort in ["pre-depression", "post-depression"]:
    for visit_session in range(1, 4):
        if mri_status == "mri":
            if visit_session == 1:
                session_column = f"1st_{depression_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{depression_cohort}_session"
            elif visit_session == 3:
                session_column = f"3rd_{depression_cohort}_session"
            elif visit_session == 4:
                session_column = f"4th_{depression_cohort}_session"
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            print(df_extracted)
            print(df_validated)
            depression_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, depression_cohort)
            depression_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, depression_cohort)
        elif mri_status == "nonmri":
            if visit_session == 1:
                session_column = f"1st_{depression_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{depression_cohort}_session"
            # elif visit_session == 3:
            #     session_column = f"3rd_{depression_cohort}_session"
            # elif visit_session == 4:
            #     session_column = f"4th_{depression_cohort}_session"
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, session_column)
            depression_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, depression_cohort)
            depression_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, depression_cohort)
            print(df_extracted)
            print(df_validated)

###############################################################################
depression_cohort = "post-depression"
df_post = data_processor.extract_post_depression_df(df_preprocessed, mri_status)
depression_save_data.save_subgroups_only_preprocessed_data(df_post, population, mri_status, depression_cohort="post-depression")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{depression_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{depression_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{depression_cohort}_session"
    df_extracted = data_processor.extract_data(df_post, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, session_column)
    print(df_extracted)
    print(df_validated)

###############################################################################
depression_cohort = "pre-depression"
df_pre = data_processor.extract_pre_depression_df(df_preprocessed, mri_status)
depression_save_data.save_subgroups_only_preprocessed_data(df_pre, population, mri_status, depression_cohort="pre-depression")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{depression_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{depression_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{depression_cohort}_session"
    df_extracted = data_processor.extract_data(df_pre, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, session_column)
    depression_save_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, depression_cohort="pre-depression")
    depression_save_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, depression_cohort="pre-depression")
    print(df_extracted)
    print(df_validated)

###############################################################################
depression_cohort = "longitudinal-depression"
df_longitudinal = data_processor.extract_longitudinal_depression_df(df_preprocessed, mri_status)
depression_save_data.save_main_preprocessed_data(df_longitudinal, population, mri_status, depression_cohort="longitudinal-depression")
for visit_session in range(1, 2):
    if visit_session == 1:
        depression_cohort = "pre-depression"
        session_column = f"1st_{depression_cohort}_session"
        df_extracted_pre = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_pre = data_processor.validate_handgrips(df_extracted_pre, session_column)
        depression_cohort = "post-depression"
        session_column = f"1st_{depression_cohort}_session"
        df_extracted_post = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_post = data_processor.validate_handgrips(df_extracted_post, session_column)   
    
    # Assuming you have DataFrames called 'df_pre' and 'df_post'
    # Concatenate the DataFrames, but keep only one of the columns with the same name
    merged_df_extracted = pd.concat([df_extracted_pre, df_extracted_post], axis=1, join="inner")
    merged_df_validated = pd.concat([df_validated_pre, df_validated_post], axis=1, join="inner")
    # Select which columns to keep (for example, keep columns from df_pre)
    df_longitudinal_extracted = merged_df_extracted.loc[:, ~merged_df_extracted.columns.duplicated()]
    df_longitudinal_validated = merged_df_validated.loc[:, ~merged_df_validated.columns.duplicated()]
    depression_cohort = "longitudinal-depression"
    session_column = f"1st_{depression_cohort}_session"
    depression_save_data.save_primary_extracted_data(df_longitudinal_extracted, population, mri_status, session_column, depression_cohort="longitudinal-depression")
    depression_save_data.save_validated_hgs_data(df_longitudinal_validated, population, mri_status, session_column, depression_cohort="longitudinal-depression")
    
print("===== Done! =====")
embed(globals(), locals())
