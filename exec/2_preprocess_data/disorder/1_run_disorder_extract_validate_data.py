
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_data.disorder import load_disorder_data
from hgsprediction.data_preprocessing import disorder_data_preprocessor
from hgsprediction.save_data.disorder import save_disorder_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
first_event = sys.argv[3]

df_original = load_disorder_data.load_original_data(population, mri_status)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

data_processor = disorder_data_preprocessor.DisorderMainDataPreprocessor(df_original, population)    
df = data_processor.define_handedness(df_original.copy())

df = data_processor.define_diagnosis_date(df)

df = data_processor.remove_missing_disorder_dates(df, first_event)

df = data_processor.remove_missing_hgs(df)

###############################################################################
if population == "stroke":
    df = data_processor.define_disorder_type(df, population)
df = data_processor.define_followup_days(df, first_event)

###############################################################################
df_preprocessed = data_processor.preprocess_disorder_df(df, population)

# Calculate and Add dominant and nondominant hgs to data
df_preprocessed = data_processor.calculate_dominant_nondominant_hgs(df_preprocessed, population)

# Remove all columns with all NaN values
df_preprocessed = data_processor.remove_nan_columns(df_preprocessed)
print("===== Done! =====")
embed(globals(), locals())
# save_disorder_data.save_main_preprocessed_data(df_preprocessed, population, mri_status, disorder_cohort=f"all-{population}-subjects", first_event=f"{first_event}")

###############################################################################
for disorder_cohort in [f"pre-{population}", f"post-{population}"]:
    for visit_session in range(1, 4):
        if mri_status == "mri":
            if first_event == "first_report":
                if visit_session == 1:
                    session_column = f"1st_{disorder_cohort}_session"
                elif visit_session == 2:
                    session_column = f"2nd_{disorder_cohort}_session"
                elif visit_session == 3:
                    session_column = f"3rd_{disorder_cohort}_session"
                    if session_column not in df_preprocessed.columns:
                        break
            elif first_event == "first_diagnosis":
                if visit_session == 1:
                    session_column = f"1st_{disorder_cohort}_session"
            # print(visit_session)
            # print(session_column)
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
            # print(df_validated)

            # save_disorder_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort, first_event)
            # save_disorder_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort, first_event)
            
        elif mri_status == "nonmri":
            if first_event == "first_report":
                if visit_session == 1:
                    session_column = f"1st_{disorder_cohort}_session"
                elif visit_session == 2:
                    session_column = f"2nd_{disorder_cohort}_session"
            elif first_event == "first_diagnosis":
                if visit_session == 1:
                    session_column = f"1st_{disorder_cohort}_session"
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
            # save_disorder_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort, first_event)
            # save_disorder_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort, first_event)
# print("===== Done! =====")
# embed(globals(), locals())
# ###############################################################################
disorder_cohort = f"post-{population}"
df_post = data_processor.extract_post_disorder_df(df_preprocessed, mri_status)
# save_disorder_data.save_subgroups_only_preprocessed_data(df_post, population, mri_status, disorder_cohort=f"post-{population}", first_event=f"{first_event}")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if first_event == "first_report":    
        if visit_session == 1:
            session_column = f"1st_{disorder_cohort}_session"
        elif visit_session == 2:
            session_column = f"2nd_{disorder_cohort}_session"
        elif visit_session == 3:
            session_column = f"3rd_{disorder_cohort}_session"
            if session_column not in df_preprocessed.columns:
                break
    elif first_event == "first_diagnosis":
        if visit_session == 1:
            session_column = f"1st_{disorder_cohort}_session"
    df_extracted = data_processor.extract_data(df_post, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
    # save_disorder_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort=f"post-{population}", first_event=f"{first_event}")
    # save_disorder_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort=f"post-{population}", first_event=f"{first_event}")
# print("===== Done! =====")
# embed(globals(), locals()) 
# ###############################################################################
disorder_cohort = f"pre-{population}"
df_pre = data_processor.extract_pre_disorder_df(df_preprocessed, mri_status)
# save_disorder_data.save_subgroups_only_preprocessed_data(df_pre, population, mri_status, disorder_cohort=f"pre-{population}", first_event=f"{first_event}")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if first_event == "first_report":    
        if visit_session == 1:
            session_column = f"1st_{disorder_cohort}_session"
        elif visit_session == 2:
            session_column = f"2nd_{disorder_cohort}_session"
        elif visit_session == 3:
            session_column = f"3rd_{disorder_cohort}_session"
    elif first_event == "first_diagnosis":
        if visit_session == 1:
            session_column = f"1st_{disorder_cohort}_session"

    df_extracted = data_processor.extract_data(df_pre, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
#     save_disorder_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort=f"pre-{population}", first_event=f"{first_event}")
#     save_disorder_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort=f"pre-{population}", first_event=f"{first_event}")
# # print("===== Done! =====")
# embed(globals(), locals())  
###############################################################################
# print("===== Done! =====")
# embed(globals(), locals())
disorder_cohort = f"longitudinal-{population}"
df_longitudinal = data_processor.extract_longitudinal_disorder_df(df_preprocessed, mri_status)
# save_disorder_data.save_main_preprocessed_data(df_longitudinal, population, mri_status, disorder_cohort=f"longitudinal-{population}", first_event=f"{first_event}")
for visit_session in range(1, 2):
    if visit_session == 1:
        disorder_cohort = f"pre-{population}"
        session_column = f"1st_{disorder_cohort}_session"
        df_extracted_pre = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_pre = data_processor.validate_handgrips(df_extracted_pre, population, session_column)
        # print("===== Done! =====")
        # embed(globals(), locals())
        disorder_cohort = f"post-{population}"
        session_column = f"1st_{disorder_cohort}_session"
        df_extracted_post = data_processor.extract_data(df_preprocessed, session_column)
        df_validated_post = data_processor.validate_handgrips(df_extracted_post, population, session_column)   
        # print("===== Done! =====")
        # embed(globals(), locals())
    # Assuming you have DataFrames called 'df_pre' and 'df_post'
    # Concatenate the DataFrames, but keep only one of the columns with the same name
    merged_df_extracted = pd.concat([df_extracted_pre, df_extracted_post], axis=1, join="inner")
    merged_df_validated = pd.concat([df_validated_pre, df_validated_post], axis=1, join="inner")
    # Select which columns to keep (for example, keep columns from df_pre)
    df_longitudinal_extracted = merged_df_extracted.loc[:, ~merged_df_extracted.columns.duplicated()]
    df_longitudinal_validated = merged_df_validated.loc[:, ~merged_df_validated.columns.duplicated()]
# print("===== Done! =====")
# embed(globals(), locals())    
    if first_event == "first_diagnosis":
        if population == "stroke":
            df_longitudinal_extracted = df_longitudinal_extracted[df_longitudinal_extracted["42007-0.0"] != 0]
            df_longitudinal_validated = df_longitudinal_validated[df_longitudinal_validated["42007-0.0"] != 0]
        elif population == "parkinson":
            df_longitudinal_extracted = df_longitudinal_extracted[df_longitudinal_extracted["42033-0.0"] != 0]
            df_longitudinal_validated = df_longitudinal_validated[df_longitudinal_validated["42033-0.0"] != 0]            
    disorder_cohort = f"longitudinal-{population}"
    session_column = f"1st_{disorder_cohort}_session"
    # save_disorder_data.save_primary_extracted_data(df_longitudinal_extracted, population, mri_status, session_column, disorder_cohort=f"longitudinal-{population}", first_event=f"{first_event}")
    # save_disorder_data.save_validated_hgs_data(df_longitudinal_validated, population, mri_status, session_column, disorder_cohort=f"longitudinal-{population}", first_event=f"{first_event}")


print("pre-dominant<4", len(df_longitudinal_extracted[df_longitudinal_extracted[f"1st_pre-{population}_hgs_dominant"]<4]))
print("pre-dominant==0", len(df_longitudinal_extracted[df_longitudinal_extracted[f"1st_pre-{population}_hgs_dominant"]==0]))
print("pre-dominant<4", len(df_longitudinal[df_longitudinal[f"1st_pre-{population}_hgs_dominant"]<4]))
print("pre-dominant<pre-nondominant", len(df_longitudinal_extracted[df_longitudinal_extracted[f"1st_pre-{population}_hgs_dominant"]<df_longitudinal_extracted[f"1st_pre-{population}_hgs_nondominant"]]))
print("pre-dominant<pre-nondominant", len(df_longitudinal[df_longitudinal[f"1st_pre-{population}_hgs_dominant"]<df_longitudinal[f"1st_pre-{population}_hgs_nondominant"]]))
print("pre-dominant<pre-nondominant validated", len(df_longitudinal_validated[df_longitudinal_validated[f"1st_pre-{population}_hgs_dominant"]<df_longitudinal_validated[f"1st_pre-{population}_hgs_nondominant"]]))
print("dominant.isna", len(df_longitudinal_extracted[df_longitudinal_extracted[f"1st_pre-{population}_hgs_dominant"].isna()]))
print("nondominant.isna", len(df_longitudinal_extracted[df_longitudinal_extracted[f"1st_pre-{population}_hgs_nondominant"].isna()]))
print("len validate",len(df_longitudinal_validated))
print("len extracted",len(df_longitudinal_extracted))

###############################################################################
print("===== Done! =====")
embed(globals(), locals())

