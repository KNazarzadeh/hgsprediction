
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
motor = sys.argv[1]
population = sys.argv[2]
mri_status = sys.argv[3]

post_list = ["1_post_session", "2_post_session", "3_post_session", "4_post_session"]
pre_list = ["1_pre_session", "2_pre_session", "3_pre_session", "4_pre_session"]

folder_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "GIT_repositories",
    "motor_ukb",
    "data_ukb",
    f"data_{motor}",
    population,
    "prepared_data",
    f"{mri_status}_{population}",
)

file_path = os.path.join(
        folder_path,
        f"{post_list[0]}_{mri_status}_{population}.csv")


# for i in range(0, len(post_list)):
#     df_name = f"df_post_{i+1}"  # Generate the data frame name dynamically
#     # Define the csv file path to save
#     file_path = os.path.join(
#         folder_path,
#         f"{post_list[i]}_{mri_status}_{population}.csv")

#     df[df_name] = pd.read_csv(file_path, sep=',')

# for i in range(0, len(pre_list)):
#     df_name = f"df_pre_{i+1}"  # Generate the data frame name dynamically
#     # Define the csv file path to save
#     file_path = os.path.join(
#         folder_path,
#         f"{pre_list[i]}_{mri_status}_{population}.csv")

#     df_name = pd.read_csv(file_path, sep=',')

df_1_post = pd.read_csv(file_path, sep=',')
df_1_post.set_index("SubjectID", inplace=True)
file_path = os.path.join(
        folder_path,
        f"available_hgs_{mri_status}_{population}.csv")

df = pd.read_csv(file_path, sep=',')
df.set_index("SubjectID", inplace=True)

# file_path = os.path.join(
#         folder_path,
#         f"{pre_list[0]}_{mri_status}_{population}.csv")
# df_1_pre = pd.read_csv(file_path, sep=',')
# df_1_pre_f = df_1_post[df_1_post['31-0.0']==0.0]
# df_1_pre_m = df_1_post[df_1_post['31-0.0']==1.0]

# filter_col = [col for col in df_1_post if col.endswith('post_session')]
# for j in range(0,4):
#     ses = df_1_post[filter_col].iloc[:,j].astype(str).str[8:]
#     for i in range(0, len(df_1_post[filter_col])):
#         idx=ses.index[i]
#         if ses.iloc[i] != "":
#             df_1_post.loc[idx, f"{j+1}_post_days"] = df_1_post.loc[idx, f"followup_days-{ses.iloc[i]}"]
#         else:
#             df_1_post.loc[idx, f"{j+1}_post_days"] = np.NaN
# days = df_1_post['followup_days-2.0'][df_1_post['followup_days-2.0']>=0]

hgs_left = "46"  # Handgrip_strength_(left)
hgs_right = "47"  # Handgrip_strength_(right)
df_tmp_mri = pd.DataFrame()
ses = 2
df_tmp_mri = df[
        ((~df[f'{hgs_left}-{ses}.0'].isna()) &
            (df[f'{hgs_left}-{ses}.0'] !=  0))
        & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
            (df[f'{hgs_right}-{ses}.0'] !=  0))]

days = df[(df['followup_days-2.0']>=0)]

df_1_post_f = days[days['31-0.0']==0.0]
df_1_post_m = days[days['31-0.0']==1.0]

f_days = df_1_post_f["followup_days-2.0"].iloc[:,0]
f_hgs_L = df_1_post_f["1_post_left_hgs"]
f_hgs_R = df_1_post_f["1_post_right_hgs"]
f_hgs_LR = df_1_post_f["1_post_left_hgs"] + df_1_post_f["1_post_right_hgs"]

m_days = df_1_post_m["followup_days-2.0"].iloc[:,0]
m_hgs_L = df_1_post_m["1_post_left_hgs"]
m_hgs_R = df_1_post_m["1_post_right_hgs"]
m_hgs_LR = df_1_post_m["1_post_left_hgs"] + df_1_post_m["1_post_right_hgs"]

df_1_post = df_1_post.loc[days.index]
df_1_post = pd.concat([df_1_post, days], axis=1)
df_1_post = df_1_post[(df_1_post["1_post_session"]=="session-1.0") |(df_1_post["1_post_session"]=="session-0.0")]

# df_1_post_f = df_1_post[df_1_post['31-0.0']==0.0]
# df_1_post_m = df_1_post[df_1_post['31-0.0']==1.0]

# f_days = df_1_post_f["followup_days-2.0"].iloc[:,0]
# f_hgs_L = df_1_post_f["1_post_left_hgs"]
# f_hgs_R = df_1_post_f["1_post_right_hgs"]
# f_hgs_LR = df_1_post_f["1_post_left_hgs"] + df_1_post_f["1_post_right_hgs"]

# m_days = df_1_post_m["followup_days-2.0"].iloc[:,0]
# m_hgs_L = df_1_post_m["1_post_left_hgs"]
# m_hgs_R = df_1_post_m["1_post_right_hgs"]
# m_hgs_LR = df_1_post_m["1_post_left_hgs"] + df_1_post_m["1_post_right_hgs"]

###############################################################################
f_corr_L, f_p_L = spearmanr(f_days, f_hgs_L)
f_corr_R, f_p_R = spearmanr(f_days, f_hgs_R)
f_corr_LR, f_p_LR = spearmanr(f_days, f_hgs_LR)

m_corr_L, m_p_L = spearmanr(m_days, m_hgs_L)
m_corr_R, m_p_R = spearmanr(m_days, m_hgs_R)
m_corr_LR, m_p_LR = spearmanr(m_days, m_hgs_LR)


###############################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
fig, ax = plt.subplots(1, 3, figsize=(30,12))
ax[0].set_box_aspect(1)
sns.set_context("poster")
sns.regplot(x=m_hgs_L, y=m_days, ax=ax[0], line_kws={"color": "red"})

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

xmax0 = xmax0+10
ymax0 = ymax0+10

ax[0].set_xlim(0, xmax0)
ax[0].set_ylim(0, ymax0)

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

yticks0 = ax[0].get_yticks()
xticks0 = ax[0].get_xticks()

text0 = 'CORR: ' + str(format(m_corr_L, '.3f'))
ax[0].set_xlabel('Actual Post Left HGS', fontsize=20, fontweight="bold")
ax[0].set_ylabel('Post-stroke days', fontsize=20, fontweight="bold")

ax[0].set_title("Post-stroke days vs Actual Post Left HGS", fontsize=15, fontweight="bold", y=1.02)
ax[0].text(xmax0 - 0.05 * xmax0, ymax0 - 0.01 * ymax0, text0, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
ax[0].plot([xmin0, xmax0], [ymin0, ymax0], 'k--')

#################################
ax[1].set_box_aspect(1)
# sns.set_context("poster")
sns.regplot(x=m_hgs_R, y=m_days, ax=ax[1], line_kws={"color": "red"})

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()
xmax1 = xmax1+10
ymax1 = ymax1+10

ax[1].set_xlim(0, xmax1)
ax[1].set_ylim(0, ymax1)

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()

yticks1 = ax[1].get_yticks()
xticks1 = ax[1].get_xticks()

text1 = 'CORR: ' + str(format(m_corr_R, '.3f'))

ax[1].set_xlabel('Actual Post Right HGS', fontsize=20, fontweight="bold")
ax[1].set_ylabel('Post-stroke days', fontsize=20, fontweight="bold")

ax[1].set_title("Post-stroke days vs Actual Post Right HGS", fontsize=15, fontweight="bold", y=1.02)
ax[1].text(xmax1 - 0.05 * xmax1, ymax1 - 0.01 * ymax1, text1, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

ax[1].plot([xmin1, xmax1], [ymin1, ymax1], 'k--')

#################################
ax[2].set_box_aspect(1)
# sns.set_context("poster")
sns.regplot(x=m_hgs_LR, y=m_days, ax=ax[2], line_kws={"color": "red"})


xmin2, xmax2 = ax[2].get_xlim()
ymin2, ymax2 = ax[2].get_ylim()

xmax2 = xmax2+10
ymax2 = ymax2+10

ax[2].set_xlim(0, xmax2)
ax[2].set_ylim(0, ymax2)

xmin2, xmax2 = ax[2].get_xlim()
ymin2, ymax2 = ax[2].get_ylim()

yticks2 = ax[2].get_yticks()
xticks2 = ax[2].get_xticks()

text2 = 'CORR: ' + str(format(m_corr_LR, '.3f'))
# ax2.xlabel('Correlation of LCOR with Actual HGS', fontsize=25, fontweight="bold")
# ax2.ylabel('Correlation of LCOR with Predicted HGS', fontsize=25, fontweight="bold")
ax[2].set_xlabel('Actual Post (Left+Right) HGS', fontsize=20, fontweight="bold")
ax[2].set_ylabel('Post-stroke days', fontsize=20, fontweight="bold")

ax[2].set_title("Post-stroke days vs Actual Post (Left+Right) HGS", fontsize=15, fontweight="bold", y=1.02)
ax[2].text(xmax2 - 0.05 * xmax2, ymax2 - 0.01 * ymax2, text2, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

ax[2].plot([xmin2, xmax2], [ymin2, ymax2], 'k--')


plt.suptitle(f"Post-stroke days vs Actual Post HGS - Males({len(df_1_post_m)})", fontsize=20, fontweight="bold", y=0.9)

plt.show()
plt.savefig(f"correlate_post_days_actual_hgs_storke_Males.png")
plt.close()
# ##############################################################################################################################################################
##############################################################################################################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
fig, ax = plt.subplots(1, 3, figsize=(30,12))
ax[0].set_box_aspect(1)
sns.set_context("poster")
sns.regplot(x=f_hgs_L, y=f_days, ax=ax[0], line_kws={"color": "red"})

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

xmax0 = xmax0+10
ymax0 = ymax0+10

ax[0].set_xlim(0, xmax0)
ax[0].set_ylim(0, ymax0)

xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()

yticks0 = ax[0].get_yticks()
xticks0 = ax[0].get_xticks()

text0 = 'CORR: ' + str(format(f_corr_L, '.3f'))
ax[0].set_xlabel('Actual Post Left HGS', fontsize=20, fontweight="bold")
ax[0].set_ylabel('Post-stroke days', fontsize=20, fontweight="bold")

ax[0].set_title("Post-stroke days vs Actual Post Left HGS", fontsize=15, fontweight="bold", y=1.02)
ax[0].text(xmax0 - 0.05 * xmax0, ymax0 - 0.01 * ymax0, text0, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

# # Add a diagonal line
ax[0].plot([xmin0, xmax0], [ymin0, ymax0], 'k--')

#################################
ax[1].set_box_aspect(1)
# sns.set_context("poster")
sns.regplot(x=f_hgs_R, y=f_days, ax=ax[1], line_kws={"color": "red"})

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()
xmax1 = xmax1+10
ymax1 = ymax1+10

ax[1].set_xlim(0, xmax1)
ax[1].set_ylim(0, ymax1)

xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()

yticks1 = ax[1].get_yticks()
xticks1 = ax[1].get_xticks()

text1 = 'CORR: ' + str(format(f_corr_R, '.3f'))

ax[1].set_xlabel('Actual Post Right HGS', fontsize=20, fontweight="bold")
ax[1].set_ylabel('Post-stroke days', fontsize=20, fontweight="bold")

ax[1].set_title("Post-stroke days vs Actual Post Right HGS", fontsize=15, fontweight="bold", y=1.02)
ax[1].text(xmax1 - 0.05 * xmax1, ymax1 - 0.01 * ymax1, text1, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

ax[1].plot([xmin1, xmax1], [ymin1, ymax1], 'k--')

#################################
ax[2].set_box_aspect(1)
# sns.set_context("poster")
sns.regplot(x=f_hgs_LR, y=f_days, ax=ax[2], line_kws={"color": "red"})


xmin2, xmax2 = ax[2].get_xlim()
ymin2, ymax2 = ax[2].get_ylim()

xmax2 = xmax2+10
ymax2 = ymax2+10

ax[2].set_xlim(0, xmax2)
ax[2].set_ylim(0, ymax2)

xmin2, xmax2 = ax[2].get_xlim()
ymin2, ymax2 = ax[2].get_ylim()

yticks2 = ax[2].get_yticks()
xticks2 = ax[2].get_xticks()

text2 = 'CORR: ' + str(format(f_corr_LR, '.3f'))
# ax2.xlabel('Correlation of LCOR with Actual HGS', fontsize=25, fontweight="bold")
# ax2.ylabel('Correlation of LCOR with Predicted HGS', fontsize=25, fontweight="bold")
ax[2].set_xlabel('Actual Post (Left+Right) HGS', fontsize=20, fontweight="bold")
ax[2].set_ylabel('Post-stroke days', fontsize=20, fontweight="bold")

ax[2].set_title("Post-stroke days vs Actual Post (Left+Right) HGS", fontsize=15, fontweight="bold", y=1.02)
ax[2].text(xmax2 - 0.05 * xmax2, ymax2 - 0.01 * ymax2, text2, verticalalignment='top',
         horizontalalignment='right', fontsize=18, fontweight="bold")

ax[2].plot([xmin2, xmax2], [ymin2, ymax2], 'k--')


plt.suptitle(f"Post-stroke days vs Actual Post HGS - Females({len(df_1_post_f)})", fontsize=20, fontweight="bold", y=0.9)

plt.show()
plt.savefig(f"correlate_post_days_actual_hgs_storke_Females.png")
plt.close()


print("===== Done! =====")
embed(globals(), locals())