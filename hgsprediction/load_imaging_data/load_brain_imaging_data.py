# Load Images data
# The images data

import os
import pandas as pd
import datatable as dt
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
def load_imaging_data(
    img_type,
    neuroanatomy,
):
    """
    The function loads image data based on the specified image type and neuroanatomy region. 
    It checks:
        - image type(GMV, LCOR, GCOR) and 
        - neuroanatomy(cortical, subcortical, cerebellum)
    to determine the appropriate file name. 
    It constructs the file path using the predefined path to the brain data directory. 
    It then loads the image data from the file into a pandas DataFrame and sets the 'SubjectID' column as the index. 
    Finally, it returns the loaded image data as a DataFrame.
    
    Parameters
    ----------
    img_type : str
             A string representing the type of image data to load:
             - "GMV" : for Grey Matter Volume
             - "GCOR": for Global Correlation
             - "LCOR": for Local Correlation
    neuroanatomy : str 
             A string representing the neuroanatomy region of interest:
             - "cortical"
             - "subcortical"
             - "cerebellum"
    Return
    -------
    img_df : DataFrame
             Containing the loaded image data.
             The index of the DataFrame is set to the 'SubjectID' column.
    """
    if neuroanatomy == "all":
        img_df = load_imaging_data_all_neuroanatomy(img_type)
    else:
        path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "brain_imaging_data",
        f"{img_type}",
        )
        
        if img_type == "GMV":
            if neuroanatomy == "cortical":
                filename = "1_gmd_schaefer_all_subjects"
            elif neuroanatomy == "subcortical":
                filename = "4_gmd_tian_all_subjects"
            elif neuroanatomy == "cerebellum":
                filename = "2_gmd_SUIT_all_subjects"
                
        elif img_type == "LCOR":
            if neuroanatomy == "cortical":
                filename = "LCOR_Schaefer400x7_Mean"
            elif neuroanatomy == "subcortical":
                filename = "LCOR_Tian_Mean"

        elif img_type == "GCOR":
            if neuroanatomy == "cortical":
                filename = "GCOR_Schaefer400x7_Mean"
            elif neuroanatomy == "subcortical":
                filename = "GCOR_Tian_Mean"

        file_path = os.path.join(path,f"{filename}.jay")

        img_dt = dt.fread(file_path)
        img_df = img_dt.to_pandas()
        img_df.set_index('SubjectID', inplace=True)

    return img_df
###############################################################################
def load_imaging_data_all_neuroanatomy(img_type):
    """
    The function loads image data based on the specified image type and neuroanatomy region. 
    It checks:
        - image type(GMV, LCOR, GCOR) and 
        - neuroanatomy(cortical, subcortical, cerebellum)
    to determine the appropriate file name. 
    It constructs the file path using the predefined path to the brain data directory. 
    It then loads the image data from the file into a pandas DataFrame and sets the 'SubjectID' column as the index. 
    Finally, it returns the loaded image data as a DataFrame.
    
    Parameters
    ----------
    img_type : str
             A string representing the type of image data to load:
             - "GMV" : for Grey Matter Volume
             - "GCOR": for Global Correlation
             - "LCOR": for Local Correlation
    neuroanatomy : str 
             A string representing the neuroanatomy region of interest:
             - "cortical"
             - "subcortical"
             - "cerebellum"
    Return
    -------
    img_df : DataFrame
             Containing the loaded image data.
             The index of the DataFrame is set to the 'SubjectID' column.
    """        
    path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "brain_imaging_data",
    f"{img_type}",
    )
    
    if img_type == "GMV":
            filename_cortical = "1_gmd_schaefer_all_subjects"
            filename_subcortical = "4_gmd_tian_all_subjects"
            filename_cerebellum = "2_gmd_SUIT_all_subjects"
            
            file_path_cortical = os.path.join(path,f"{filename_cortical}.jay")
            file_path_subcortical = os.path.join(path,f"{filename_subcortical}.jay")
            file_path_cerebellum = os.path.join(path,f"{filename_cerebellum}.jay")
            
            features_cortical = dt.fread(file_path_cortical).to_pandas()
            features_subcortical = dt.fread(file_path_subcortical).to_pandas()
            features_cerebellum = dt.fread(file_path_cerebellum).to_pandas()

            features_cortical.set_index('SubjectID', inplace=True)
            features_subcortical.set_index('SubjectID', inplace=True)
            features_cerebellum.set_index('SubjectID', inplace=True)
            
            img_df = pd.concat([features_cortical, features_subcortical, features_cerebellum], axis=1)
            
    elif img_type == "LCOR":
            filename_cortical = "LCOR_Schaefer400x7_Mean"
            filename_subcortical = "LCOR_Tian_Mean"
            
            file_path_cortical = os.path.join(path,f"{filename_cortical}.jay")
            file_path_subcortical = os.path.join(path,f"{filename_subcortical}.jay")
            
            features_cortical = dt.fread(file_path_cortical).to_pandas()
            features_subcortical = dt.fread(file_path_subcortical).to_pandas()
            
            features_cortical.set_index('SubjectID', inplace=True)
            features_subcortical.set_index('SubjectID', inplace=True)
                        
            img_df = pd.concat([features_cortical, features_subcortical], axis=1)
            
    elif img_type == "GCOR":
            filename_cortical = "GCOR_Schaefer400x7_Mean"
            filename_subcortical = "GCOR_Tian_Mean"
            
            file_path_cortical = os.path.join(path,f"{filename_cortical}.jay")
            file_path_subcortical = os.path.join(path,f"{filename_subcortical}.jay")
            
            features_cortical = dt.fread(file_path_cortical).to_pandas()
            features_subcortical = dt.fread(file_path_subcortical).to_pandas()

            features_cortical.set_index('SubjectID', inplace=True)
            features_subcortical.set_index('SubjectID', inplace=True)
            
            img_df = pd.concat([features_cortical, features_subcortical], axis=1)
    
    
    img_df.index = img_df.index.str.replace("sub-", "")
    # print("===== Done! =====")
    # embed(globals(), locals())
    # img_df.index = img_df.index.map(int)
    
    img_df = img_df.dropna()

    return img_df
