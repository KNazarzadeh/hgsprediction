# Load Images data
# The images data

import os
import datatable as dt

###############################################################################
def load_image_data(
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
    elif img_type == "LCOR":
        if neuroanatomy == "cortical":
            filename = "GCOR_Schaefer400x7_Mean"
        elif neuroanatomy == "subcortical":
            filename = "GCOR_Tian_Mean"

    path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "brain_data",
    )
                
    file_path = os.path.join(path,f"{filename}.jay")
    

    img_dt = dt.fread(file_path)
    img_df = img_dt.to_pandas()
    img_df.set_index('SubjectID', inplace=True)

    return img_df
