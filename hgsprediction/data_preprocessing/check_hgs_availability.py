
import pandas as pd

def check_hgs_availability(df):
    
    # Handgrip strength info
    # for Left and Right Hands
    hgs_left = "46"  # Handgrip_strength_(left)
    hgs_right = "47"  # Handgrip_strength_(right)
    # UK Biobank assessed handgrip strength in 4 sessions
    session = 4 # 0 to 3
    
    for ses in range(session):
        df_tmp = df[
            ((~df[f'{hgs_left}-{ses}.0'].isna()) &
             (df[f'{hgs_left}-{ses}.0'] !=  0))
            & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
               (df[f'{hgs_right}-{ses}.0'] !=  0))
        ]
        df_output = pd.concat([df_output, df_tmp])

    # Drop the duplicated subjects
    # based on 'eid' column (subject ID)
    df_output = df_output.drop_duplicates(subset=['Subjectid'], keep="first")

    return df