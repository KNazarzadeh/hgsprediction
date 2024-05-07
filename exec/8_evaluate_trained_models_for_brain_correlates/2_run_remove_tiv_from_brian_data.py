
import pandas as pd
import numpy as np
import os
import sys
import datatable as dt

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
brain_data_type = sys.argv[1]
schaefer = sys.argv[2]

###############################################################################
jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    f"{brain_data_type.upper()}",
)

schaefer_file = os.path.join(jay_path, f"{brain_data_type.upper()}_Schaefer{schaefer}x7_Mean.jay")

dt_schaefer = dt.fread(schaefer_file)
df_schaefer = dt_schaefer.to_pandas()
df_schaefer.set_index('SubjectID', inplace=True)

tian_file = os.path.join(jay_path, f"4_gmd_tianS1_all_subjects.jay")
dt_tian = dt.fread(tian_file)
df_tian = dt_tian.to_pandas()
df_tian.set_index('SubjectID', inplace=True)

if brain_data_type == "gmv":
    suit_file = os.path.join(jay_path, f"{brain_data_type.upper()}_SUIT_Mean.jay")
    dt_suit = dt.fread(suit_file)
    df_suit = dt_suit.to_pandas()
    df_suit.set_index('SubjectID', inplace=True)

merged_df = pd.merge(df_schaefer, df_tian, left_index=True, right_index=True, how='inner')
merged_df = pd.merge(merged_df, df_suit, left_index=True, right_index=True, how='inner')

merged_df = merged_df.dropna()
merged_df.index = merged_df.index.str.replace("sub-", "")
merged_df.index = merged_df.index.map(int)

tiv_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    f"TIV",
)

df_tiv = pd.read_csv(f"{tiv_path}/cat_rois_Schaefer2018_600Parcels_17Networks_order.csv", sep=',', index_col=0)
print("===== Done! =====")
embed(globals(), locals())
##############################################################################