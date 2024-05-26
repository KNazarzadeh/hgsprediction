import math
import sys
import os
import numpy as np
import pandas as pd


from ptpython.repl import embed



all_file_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/all_healthy/all_healthy.csv"

mri_file_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/mri_healthy/mri_healthy.csv"

nonmri_file_path = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/healthy/original_data/nonmri_healthy/nonmri_healthy.csv"

df_all = pd.read_csv(all_file_path, sep=',', low_memory=False)

df_mri = pd.read_csv(mri_file_path, sep=',', low_memory=False)

df_nonmri = pd.read_csv(nonmri_file_path, sep=',', low_memory=False)

df_all = df_all.rename(columns={"eid":"SubjectID"})
df_all = df_all.set_index("SubjectID")

df_mri = df_mri.rename(columns={"eid":"SubjectID"})
df_mri = df_mri.set_index("SubjectID")

df_nonmri = df_nonmri.rename(columns={"eid":"SubjectID"})
df_nonmri = df_nonmri.set_index("SubjectID")


###################################################################
# Count the number of female participants in the ALL data
print("Female, ALL data (MRI and non-MRI)=", len(df_all[df_all['31-0.0']==0]))

# Calculate the percentage of female participants in the ALL data (MRI and non-MRI)
print("Female %, ALL data (MRI and non-MRI)=", "{:.2f}".format(len(df_all[df_all['31-0.0']==0])*100/len(df_all)))

# Count the number of male participants in the ALL data (MRI and non-MRI)
print("Male, ALL data (MRI and non-MRI)=", len(df_all[df_all['31-0.0']==1]))

# Calculate the percentage of male participants in the ALL data (MRI and non-MRI)
print("Male %, ALL data (MRI and non-MRI)=", "{:.2f}".format(len(df_all[df_all['31-0.0']==1])*100/len(df_all)))

# Compute summary statistics for the ALL data (MRI and non-MRI) and round to two decimal places
summary_stats_all = df_all.describe().apply(lambda x: round(x, 2))

# Print the mean and standard deviation of the 'Age1stVisit' column in the ALL data (MRI and non-MRI)
print("Age \n=", summary_stats_all['Age1stVisit'])

# Print the summary statistics for the ALL data (MRI and non-MRI) data
print("ALL data (MRI and non-MRI) data describe=\n", summary_stats_all)

Print("Age, BMI, Height, wasit-to-hip ration, HGS left, HGS right")
print("All Females:\n:", df_all[df_all['31-0.0']==0].describe().apply(lambda x: round(x, 2)))

###################################################################
# Count the number of female participants in the non-MRI data
print("Female, non-MRI=", len(df_nonmri[df_nonmri['31-0.0']==0]))

# Calculate the percentage of female participants in the non-MRI data
print("Female %, non-MRI=", "{:.2f}".format(len(df_nonmri[df_nonmri['31-0.0']==0])*100/len(df_nonmri)))

# Count the number of male participants in the non-MRI data
print("Male, non-MRI=", len(df_nonmri[df_nonmri['31-0.0']==1]))

# Calculate the percentage of male participants in the non-MRI data
print("Male %, non-MRI=", "{:.2f}".format(len(df_nonmri[df_nonmri['31-0.0']==1])*100/len(df_nonmri)))

# Compute summary statistics for the non-MRI data and round to two decimal places
summary_stats_nonmri = df_nonmri.describe().apply(lambda x: round(x, 2))

# Print the mean and standard deviation of the 'Age1stVisit' column in the non-MRI data
print("Age =\n", summary_stats_nonmri['Age1stVisit'])

# Print the summary statistics for the non-MRI data
print("non-MRI data describe=\n", summary_stats_nonmri)
###################################################################

# Count the number of female participants in the MRI data
print("Female, MRI=", len(df_mri[df_mri['31-0.0']==0]))

# Calculate the percentage of female participants in the MRI data
print("Female %, MRI=", "{:.2f}".format(len(df_mri[df_mri['31-0.0']==0])*100/len(df_mri)))

# Count the number of male participants in the MRI data
print("Male, MRI=", len(df_mri[df_mri['31-0.0']==1]))

# Calculate the percentage of male participants in the MRI data
print("Male %, MRI=", "{:.2f}".format(len(df_mri[df_mri['31-0.0']==1])*100/len(df_mri)))

# Compute summary statistics for the MRI data and round to two decimal places
summary_stats_mri = df_mri.describe().apply(lambda x: round(x, 2))

# Print the mean and standard deviation of the 'Age1stVisit' column in the MRI data
print("Age =\n", summary_stats_mri['Age1stVisit'])

# Print the summary statistics for the MRI data
print("MRI data describe=\n", summary_stats_mri)


print("===== Done! =====")
embed(globals(), locals())