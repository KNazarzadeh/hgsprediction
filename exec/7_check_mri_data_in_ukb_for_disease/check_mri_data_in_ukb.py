import os
import pandas as pd
import tempfile
import datalad.api as dl
import shutil
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


stroke_folder = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/stroke/original_data/mri_stroke/mri_stroke.csv"
longitudinal_folder = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/stroke/preprocessed_data/mri_stroke/longitudinal-stroke_data/1st_longitudinal-stroke_session_data/preprocessed_data/1st_longitudinal-stroke_session_preprocessed_data.csv"
post_only = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/stroke/preprocessed_data/mri_stroke/post-stroke_data/1st_post-stroke_session_data/preprocessed_data/1st_post-stroke_session_preprocessed_data.csv"
pre_only = "/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/data_hgs/stroke/preprocessed_data/mri_stroke/pre-stroke_data/1st_pre-stroke_session_data/preprocessed_data/1st_pre-stroke_session_preprocessed_data.csv"

df_all = pd.read_csv(stroke_folder, sep=',', index_col=0)
df_longitudinal = pd.read_csv(longitudinal_folder, sep=',', index_col=0)
df_post_only = pd.read_csv(post_only, sep=',', index_col=0)
# print("===== Done! =====")
# embed(globals(), locals()) 
subjects = [str(idx) for idx in df_longitudinal.index]
# -----------------------------------------------------
# -- Define the list of subjects
# -----------------------------------------------------
# subj_IDs = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
subj_IDs = ["sub-" + item for item in subjects]
# subj_IDs = subjects
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
folder_names = 'anat'
# -----------------------------------------------------
# -- Loop over the subjects
# -----------------------------------------------------
subjs_with_mri = []
for subj_ID in subj_IDs:
    subj_folder = os.path.join(
        database_folder, subj_ID)
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
        if os.path.exists(os.path.join(mri_folder, 'anat')):
            anat_folder = os.path.join(mri_folder, 'anat')
            os.chdir(anat_folder)
            # Specify the name of the file you want to find
            file_to_find = f"{subj_ID}_ses-2_T1w.nii.gz"
            # Use os.listdir() to list all items in the directory
            items = os.listdir(anat_folder)
            # Check if the file exists in the directory
            if file_to_find in items:
                subjs_with_mri.append(subj_ID)
                dl.get(file_to_find)
                t1_folder_path = os.path.join(anat_folder, file_to_find)
                mri_stroke_folder = "/data/project/stroke_ukb/knazarzadeh/data_ukk/tmp/mri_stroke/longitudinal-stroke/"
                shutil.copy(t1_folder_path, mri_stroke_folder)
                
            else:
                print(f"{file_to_find} was not found in the directory.")
                
        else:
            print(f"The folder anat was not found in the directory.")

        os.chdir(database_folder)
    else:
        print(f"The folder '{subj_ID}' does not exist.")
        i=i+1
        print(i)
        print(subj_ID)

print("===== Done! =====")
embed(globals(), locals())    

      