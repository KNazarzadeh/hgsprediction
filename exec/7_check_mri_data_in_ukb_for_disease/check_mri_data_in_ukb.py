import os
import pandas as pd
import tempfile
import datalad.api as dl
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


stroke_folder = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/stroke/original_data/mri_stroke/mri_stroke.csv"
longitudinal_folder = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/stroke/preprocessed_data/mri_stroke/longitudinal-stroke_data/1st_longitudinal-stroke_session_data/preprocessed_data/1st_longitudinal-stroke_session_preprocessed_data.csv"
post_only = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/stroke/preprocessed_data/mri_stroke/post-stroke_data/1st_post-stroke_session_data/preprocessed_data/1st_post-stroke_session_preprocessed_data.csv"

df_all = pd.read_csv(stroke_folder, sep=',', index_col=0)
df_longitudinal = pd.read_csv(longitudinal_folder, sep=',', index_col=0)
df_post_only = pd.read_csv(post_only, sep=',', index_col=0)

subjects = [str(idx) for idx in df.index]
# -----------------------------------------------------
# -- Define the list of subjects
# -----------------------------------------------------
# subj_IDs = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
# subj_IDs = ["sub-" + item for item in subjects]
subj_IDs = subjects
i=0
# -----------------------------------------------------
# -- Clone the ukb dataset usign datalad
# -----------------------------------------------------
# repo = "ria+http://ukb.ds.inm7.de#~bids"
# database_folder = tempfile.mkdtemp()
# dl.install(database_folder, source=repo)
# dataset = dl.Dataset(database_folder)

database_folder = "/data/project/stroke_ukb/knazarzadeh/data_ukk/tmp/mri_bids"
os.chdir(database_folder)

# List of folder names to check for
folder_names = ['swi', 'anat', 'non-bids', 'func']
# -----------------------------------------------------
# -- Loop over the subjects
# -----------------------------------------------------
subjs_with_mri = []
for subj_ID in subj_IDs:
    subj_folder = os.path.join(
        database_folder, 
        'sub-' + str(subj_ID)
    )
    # Check if the folder exists
    if os.path.exists(subj_folder):
        # Change the current working directory to the specified folder
        # -----------------------------------------------------
        # -- Get the data for the subject
        # ----------------------------------------------------- 
        dl.get(subj_folder, get_data=False)
        mri_folder = os.path.join(subj_folder, "ses-2")
        os.chdir(mri_folder)
           
        # ----- Get a single rsfMRI dataset for further analysis
        # Check if any of the specified folders exist in the directory
        existing_folders = [folder for folder in folder_names if os.path.exists(os.path.join(mri_folder, folder))]

        subjs_with_mri.append(subj_ID)
        os.chdir(database_folder)

    else:
        print(f"The folder '{subj_ID}' does not exist.")
        i=i+1
        print(i)
    

print("===== Done! =====")
embed(globals(), locals())    

      