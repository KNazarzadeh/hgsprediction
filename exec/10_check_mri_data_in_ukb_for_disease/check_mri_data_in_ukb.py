import sys
import os
import pandas as pd
from hgsprediction.load_data.disorder import load_disorder_data
from datalad.api import get, drop

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
disorder_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
model_name = sys.argv[7]
confound_status = sys.argv[8]
n_repeats = sys.argv[9]
n_folds = sys.argv[10]
gender = sys.argv[11]
first_event = sys.argv[12]
session = sys.argv[13]
##################################################
# for all session pre- and -post disorder together (all in one):
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##################################################
# Load the extracted data based on various parameters
df = load_disorder_data.load_extracted_data_by_feature_and_target(
        population,
        mri_status,
        session_column,
        feature_type,
        target,
        gender,
        first_event,
    )
##################################################
subjects_long = [str(idx) for idx in df[df['1st_post-stroke_session']== float(f"{session}.0")].index]

# print("===== Done! =====")
# embed(globals(), locals()) 

# -----------------------------------------------------
# -- Define the list of subjects
# -----------------------------------------------------
# subj_IDs = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
subj_IDs = ["sub-" + item for item in subjects_long]
# subj_IDs = subjects
i=0
# -----------------------------------------------------
# -- Clone the ukb dataset usign datalad
# -----------------------------------------------------
# repo = "ria+http://ukb.ds.inm7.de#~bids"
# database_folder = tempfile.mkdtemp()
# dl.install(database_folder, source=repo)
# dataset = dl.Dataset(database_folder)

database_folder = "/data/project/stroke_ukb/knazarzadeh/ukb_original_data/new_version_ukb_bids_2024/ukb_bids/"
os.chdir(database_folder)

# List of folder names to check for
folder_names = 'anat'
# print("===== Done! =====")
# embed(globals(), locals()) 
# -----------------------------------------------------
# -- Loop over the subjects
# -----------------------------------------------------
subjs_with_mri = []
for subj_ID in subj_IDs:
    subj_folder = os.path.join(database_folder, subj_ID)
    print(f"i={i}, subj_ID={subj_ID}")
    # Check if the folder exists
    if os.path.exists(subj_folder):
        # Change the current working directory to the specified folder
        # -----------------------------------------------------
        # -- Get the data for the subject
        # ----------------------------------------------------- 
        get(subj_folder, get_data=False)       
        # print("===== Done! =====")
        # embed(globals(), locals())  
        if os.path.exists(os.path.join(subj_folder, f"ses-{session}")):
            mri_folder = os.path.join(subj_folder, f"ses-{session}")
            os.chdir(mri_folder)
            # ----- Get a single rsfMRI dataset for further analysis
            # Check if any of the specified folders exist in the directory
            if os.path.exists(os.path.join(mri_folder, 'anat')):
                anat_folder = os.path.join(mri_folder, 'anat')
                os.chdir(anat_folder)
                # Specify the name of the file you want to find
                file_to_find = f"{subj_ID}_ses-{session}_T1w.nii.gz"
                # Use os.listdir() to list all items in the directory
                items = os.listdir(anat_folder)
                # Check if the file exists in the directory
                if file_to_find in items:
                    subjs_with_mri.append(subj_ID)
                    # get(file_to_find)
                    # t1_folder_path = os.path.join(anat_folder, file_to_find)
                    # mri_stroke_folder = "/data/project/stroke_ukb/knazarzadeh/data_ukk/tmp/mri_stroke/longitudinal-stroke_T1w/"
                    # shutil.copy(t1_folder_path, mri_stroke_folder)
                    # dl.drop(file_to_find)
                else:
                    print(f"{file_to_find} was not found in the directory.")
                    
            else:
                print(f"The anat was not found in the {mri_folder}.")

            os.chdir(database_folder)
            drop(subj_folder)
        else:
            print(f"The folder '{subj_ID}' does not exist.")
        
    i = i+1

modified_list = [s.replace("sub-", "") for s in subjs_with_mri]
int_list = [int(item) for item in modified_list]

print("===== Done! =====")
embed(globals(), locals())    

      