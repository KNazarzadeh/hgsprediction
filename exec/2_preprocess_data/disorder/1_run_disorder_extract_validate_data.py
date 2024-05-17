
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_data import disorder_load_data
from hgsprediction.data_preprocessing import disorder_data_preprocessor
from hgsprediction.save_data import disorder_save_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

df_original = disorder_load_data.load_original_data(population, mri_status)

###############################################################################

data_processor = disorder_data_preprocessor.DisorderMainDataPreprocessor(df_original, population)    
df = data_processor.define_handness(df_original)


icd10_code = "G20"

# def process_data_for_icd10_codes(df, icd10_codes):
#     for icd10_code in icd10_codes:
#         filter_rows = df[df.filter(like="41270").astype(str).apply(lambda x: x.str.startswith(icd10_code)).any(axis=1)]
        
#         filtered_columns_first_diagnoses_code = filter_rows.filter(regex=("41270"))
#         filtered_columns_first_diagnoses_code = filtered_columns_first_diagnoses_code.dropna(axis=1, how="all")
        
#         result = filtered_columns_first_diagnoses_code.apply(lambda row: next((col for col in row.index if str(row[col]).startswith(icd10_code)), None), axis=1)

#         first_diagnoses_code_column_name = f"first_diagnoses_code_column_{icd10_code}"
#         first_diagnoses_date_column_name = f"first_diagnoses_date_column_{icd10_code}"
#         first_diagnoses_date_name = f"first_diagnoses_date_{icd10_code}"

#         df[first_diagnoses_code_column_name] = np.nan
#         df[first_diagnoses_date_column_name] = np.nan
#         df[first_diagnoses_date_name] = np.nan
        
#         for idx in df.index:
#             if idx in result.index:
#                 df.loc[idx, first_diagnoses_code_column_name] = str(result.loc[idx])
        
#         filtered_columns_first_diagnoses_date = filter_rows.filter(regex=("41280"))
#         filtered_columns_first_diagnoses_date = filtered_columns_first_diagnoses_date.dropna(axis=1, how="all")
#         filtered_columns_first_diagnoses_date = filtered_columns_first_diagnoses_date.reindex(result.index)

#         for idx in df.index:
#             if pd.notna(df.loc[idx, first_diagnoses_code_column_name]):
#                 df_diagnoses_code_split = df.loc[idx, first_diagnoses_code_column_name].split('-')[1]
#                 if idx in filtered_columns_first_diagnoses_date.index:
#                     col_diagnoses_date = [col for col in filtered_columns_first_diagnoses_date.columns if df_diagnoses_code_split in col]
#                     if col_diagnoses_date:
#                         df.loc[idx, first_diagnoses_date_column_name] = col_diagnoses_date[0]
#                         df.loc[idx, first_diagnoses_date_name] = filtered_columns_first_diagnoses_date.loc[idx, col_diagnoses_date[0]]

#     return df

# Example usage:
# df = ... # Load or create your DataFrame
# icd10_codes = ["G20", "F32", "I10"]
# df = process_data_for_icd10_codes(df, icd10_codes)

filter_rows = df[df.filter(like="41270").astype(str).apply(lambda x: x.str.startswith(icd10_code)).any(axis=1)]
filtered_columns_first_diagnoses_code = filter_rows.filter(regex=("41270"))
filtered_columns_first_diagnoses_code = filtered_columns_first_diagnoses_code.dropna(axis=1, how="all")
result = filtered_columns_first_diagnoses_code.apply(lambda row: next((col for col in row.index if str(row[col]).startswith(icd10_code)), None), axis=1)

for idx in df.index:
    if idx in result.index:            
        df.loc[idx, "first_diagnoses_code_column"] = str(result.loc[idx])
    else:
        df.loc[idx, "first_diagnoses_code_column"] = np.NaN


filtered_columns_first_diagnoses_date = filter_rows.filter(regex=("41280"))
filtered_columns_first_diagnoses_date = filtered_columns_first_diagnoses_date.dropna(axis=1, how="all")
filtered_columns_first_diagnoses_date = filtered_columns_first_diagnoses_date.reindex(result.index)

# Step 1: Extract the second part of split values from column_x in df2
for idx in df.index:
    if pd.notna(df.loc[idx, "first_diagnoses_code_column"]):
        df_diagnoses_code_split = df.loc[idx, "first_diagnoses_code_column"].split('-')[1]
        if idx in filtered_columns_first_diagnoses_date.index:     
            col_diagnoses_date = [col for col in filtered_columns_first_diagnoses_date.columns if df_diagnoses_code_split in col]
            df.loc[idx, "first_diagnoses_date_column"] = col_diagnoses_date[0]
            df.loc[idx, "first_diagnoses_date"] = filtered_columns_first_diagnoses_date.loc[idx, col_diagnoses_date[0]]

    else:
        df.loc[idx, "first_diagnoses_date_column"] = np.NaN
        df.loc[idx, "first_diagnoses_date"] = np.NaN


a = df[(~df['42032-0.0'].isna()) & (~df['first_diagnoses_date'].isna())]
a[['131022-0.0', '42032-0.0', 'first_diagnoses_date']]
print("===== Done! =====")
embed(globals(), locals())

df = data_processor.remove_missing_disorder_dates(df, population)

df = data_processor.remove_missing_hgs(df)

###############################################################################
if population == "stroke":
    df = data_processor.define_disorder_type(df, population)
df = data_processor.define_followup_days(df, population)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df_preprocessed = data_processor.preprocess_disorder_df(df, population)
# Calculate and Add dominant and nondominant hgs to data
df_preprocessed = data_processor.calculate_dominant_nondominant_hgs(df_preprocessed, population)
# Remove all columns with all NaN values
df_preprocessed = data_processor.remove_nan_columns(df_preprocessed)

disorder_save_data.save_main_preprocessed_data(df_preprocessed, population, mri_status, disorder_cohort=f"all-{population}-subjects")

###############################################################################
for disorder_cohort in [f"pre-{population}", f"post-{population}"]:
    for visit_session in range(1, 4):
        if mri_status == "mri":
            if visit_session == 1:
                session_column = f"1st_{disorder_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{disorder_cohort}_session"
            elif visit_session == 3:
                session_column = f"3rd_{disorder_cohort}_session"
                if session_column not in df_preprocessed.columns:
                    break
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
            disorder_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort)
            disorder_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort)
            
        elif mri_status == "nonmri":
            if visit_session == 1:
                session_column = f"1st_{disorder_cohort}_session"
            elif visit_session == 2:
                session_column = f"2nd_{disorder_cohort}_session"
            df_extracted = data_processor.extract_data(df_preprocessed, session_column)
            df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
            disorder_save_data.save_primary_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort)
            disorder_save_data.save_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort)
# ###############################################################################
disorder_cohort = f"post-{population}"
df_post = data_processor.extract_post_disorder_df(df_preprocessed, mri_status)
disorder_save_data.save_subgroups_only_preprocessed_data(df_post, population, mri_status, disorder_cohort=f"post-{population}")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{disorder_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{disorder_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{disorder_cohort}_session"
        if session_column not in df_preprocessed.columns:
            break
    df_extracted = data_processor.extract_data(df_post, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
    disorder_save_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort=f"post-{population}")
    disorder_save_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort=f"post-{population}")
# ###############################################################################
disorder_cohort = f"pre-{population}"
df_pre = data_processor.extract_pre_disorder_df(df_preprocessed, mri_status)
disorder_save_data.save_subgroups_only_preprocessed_data(df_pre, population, mri_status, disorder_cohort=f"pre-{population}")
if mri_status == "mri":
    visit_range = range(1, 4)
elif mri_status == "nonmri":
    visit_range = range(1, 3)
for visit_session in visit_range:
    if visit_session == 1:
        session_column = f"1st_{disorder_cohort}_session"
    elif visit_session == 2:
        session_column = f"2nd_{disorder_cohort}_session"
    elif visit_session == 3:
        session_column = f"3rd_{disorder_cohort}_session"
    df_extracted = data_processor.extract_data(df_pre, session_column)
    df_validated = data_processor.validate_handgrips(df_extracted, population, session_column)
    disorder_save_data.save_subgroups_only_extracted_data(df_extracted, population, mri_status, session_column, disorder_cohort=f"pre-{population}")
    disorder_save_data.save_subgroups_only_validated_hgs_data(df_validated, population, mri_status, session_column, disorder_cohort=f"pre-{population}")

###############################################################################
disorder_cohort = f"longitudinal-{population}"
df_longitudinal = data_processor.extract_longitudinal_disorder_df(df_preprocessed, mri_status)
disorder_save_data.save_main_preprocessed_data(df_longitudinal, population, mri_status, disorder_cohort=f"longitudinal-{population}")
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
    disorder_cohort = f"longitudinal-{population}"
    session_column = f"1st_{disorder_cohort}_session"
    disorder_save_data.save_primary_extracted_data(df_longitudinal_extracted, population, mri_status, session_column, disorder_cohort=f"longitudinal-{population}")
    disorder_save_data.save_validated_hgs_data(df_longitudinal_validated, population, mri_status, session_column, disorder_cohort=f"longitudinal-{population}")


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

