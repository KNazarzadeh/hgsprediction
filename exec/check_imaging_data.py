import os
import pandas as pd
import datatable as dt
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
# Specify the directory path
# folder_path = os.path.join('/data/project/stroke_ukb/knazarzadeh/data_ukk/tmp/ukb_fc_metrix/LCOR/')

jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "brain_data",
)

jay_lcor = os.path.join(
        jay_path,
        'LCOR_Schaefer400x7_Mean.jay')

jay_gcor = os.path.join(
        jay_path,
        'GCOR_Schaefer400x7_Mean.jay')

jay_file_1 = os.path.join(
        jay_path,
        '1_gmd_schaefer_all_subjects.jay')
jay_file_4 = os.path.join(
        jay_path,
        '4_gmd_tian_all_subjects.jay')
jay_file_2 = os.path.join(
        jay_path,
        '2_gmd_SUIT_all_subjects.jay')

# fname = base_dir / '1_gmd_schaefer_all_subjects.jay'
# feature_dt = dt.fread(jay_file.as_posix())
feature_dt_1 = dt.fread(jay_file_1)
feature_df_1 = feature_dt_1.to_pandas()
feature_df_1.set_index('SubjectID', inplace=True)

feature_dt_4 = dt.fread(jay_file_4)
feature_df_4 = feature_dt_4.to_pandas()
feature_df_4.set_index('SubjectID', inplace=True)

feature_dt_2 = dt.fread(jay_file_2)
feature_df_2 = feature_dt_2.to_pandas()
feature_df_2.set_index('SubjectID', inplace=True)

feature_df_1.index = feature_df_1.index.str.replace("sub-", "")
feature_df_1.index = feature_df_1.index.map(int)

feature_df_2.index = feature_df_2.index.str.replace("sub-", "")
feature_df_2.index = feature_df_2.index.map(int)

feature_df_4.index = feature_df_4.index.str.replace("sub-", "")
feature_df_4.index = feature_df_4.index.map(int)

feature_dt_lcor = dt.fread(jay_lcor)
feature_df_lcor = feature_dt_4.to_pandas()
feature_df_lcor.set_index('SubjectID', inplace=True)

feature_dt_gcor = dt.fread(jay_gcor)
feature_df_gcor = feature_dt_2.to_pandas()
feature_df_gcor.set_index('SubjectID', inplace=True)

feature_df_lcor.index = feature_df_lcor.index.str.replace("sub-", "")
feature_df_lcor.index = feature_df_lcor.index.map(int)

feature_df_gcor.index = feature_df_gcor.index.str.replace("sub-", "")
feature_df_gcor.index = feature_df_gcor.index.map(int)


df = pd.read_csv("stroke.csv", sep=',')
df_mri = df[(df['followup_days-2.0']>=0) | (df['followup_days-3.0']>=0)]

stroke_gcor = df_mri[df_mri['SubjectID'].isin(feature_df_gcor.index)]
stroke_lcor = df_mri[df_mri['SubjectID'].isin(feature_df_lcor.index)]

stroke_gm1 = df_mri[df_mri['SubjectID'].isin(feature_df_1.index)]
stroke_gm2 = df_mri[df_mri['SubjectID'].isin(feature_df_2.index)]
stroke_gm4 = df_mri[df_mri['SubjectID'].isin(feature_df_4.index)]

print("===== Done! =====")
embed(globals(), locals())



