
import os
from ptpython.repl import embed





folder_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "data_ukk",
    "tmp",
    "bids",
)

# Get a list of all files in the folder
subject_list = os.listdir(folder_path)

# Count the number of files
# Iterate over subfolders within the main folder
for subject_name in subject_list:
    subfolder_path = os.path.join(folder_path, subject_name)
    if os.path.isdir(subfolder_path):
        print(subject_name)
    # if "ses-2" in os.listdir(subject_name):sub
        # Perform operations on the subfolder
        print("Processing subfolder:", subject_name)
        
        
print("===== Done! =====")
embed(globals(), locals())
