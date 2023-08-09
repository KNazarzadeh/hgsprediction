
import os
import pandas as pd
import numpy as np

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

#--------------------------------------------------------------------------#
def prepare_stroke(target):
    post_list = ["1_post_session", "2_post_session", "3_post_session", "4_post_session"]
    motor = "hgs"
    mri_status = "mri"
    population = "stroke"

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

    df_post = pd.read_csv(file_path, sep=',')

    df_post.set_index("SubjectID", inplace=True)

    ses = df_post["1_post_session"].astype(str).str[8:]
    for i in range(0, len(df_post["1_post_session"])):
        idx=ses.index[i]
        if ses.iloc[i] != "":
            df_post.loc[idx, "1_post_days"] = df_post.loc[idx, f"followup_days-{ses.iloc[i]}"]
        else:
            df_post.loc[idx, "1_post_days"] = np.NaN

    ##############################################################################
    df_ses3 = df_post[df_post["1_post_session"] == "session-3.0"]
    df_ses2 = df_post[~df_post.index.isin(df_ses3.index)]
    # Replace Age
    # df_ses3.loc[:, 'post_age'] = df_ses3.loc[:, f'21003-3.0']
    df_ses3.loc[:, 'post_age'] = df_ses3.loc[:, f'AgeAt2ndScan']

    ##############################################################################
    # Replace BMI
    df_ses3.loc[:, 'post_bmi'] = df_ses3.loc[:, f'21001-3.0']
    ##############################################################################
    # Replace Height
    df_ses3.loc[:, 'post_height'] = df_ses3.loc[:, f'50-3.0']

    df_ses3.loc[:, 'post_days'] = df_ses3.loc[:, f'followup_days-3.0']

    ##############################################################################
    # Replace waist to hip ratio
    df_ses3.loc[:, 'post_waist'] = df_ses3.loc[:, f'48-3.0']
    df_ses3.loc[:, 'post_hip'] = df_ses3.loc[:, f'49-3.0']

    df_ses3['post_waist_hip_ratio'] = (df_ses3.loc[:, "post_waist"].astype(str).astype(float)).div(
                    df_ses3.loc[:, "post_hip"].astype(str).astype(float))
    ##############################################################################
    # Replace Age
    # df_ses2.loc[:, 'post_age'] = df_ses2.loc[:, f'21003-2.0']
    df_ses2.loc[:, 'post_age'] = df_ses2.loc[:, f'AgeAtScan']

    ##############################################################################
    # Replace BMI
    df_ses2.loc[:, 'post_bmi'] = df_ses2.loc[:, f'21001-2.0']
    ##############################################################################
    # Replace Height
    df_ses2.loc[:, 'post_height'] = df_ses2.loc[:, f'50-2.0']

    df_ses2.loc[:, 'post_days'] = df_ses2.loc[:, f'followup_days-2.0']

    ##############################################################################
    # Replace waist to hip ratio
    df_ses2.loc[:, 'post_waist'] = df_ses2.loc[:, f'48-2.0']
    df_ses2.loc[:, 'post_hip'] = df_ses2.loc[:, f'49-2.0']

    df_ses2['post_waist_hip_ratio'] = (df_ses2.loc[:, "post_waist"].astype(str).astype(float)).div(
                    df_ses2.loc[:, "post_hip"].astype(str).astype(float))
    ##############################################################################
    sub_id = df_ses2[df_ses2['1707-0.0']== 1.0].index.values
    # Add and new column "dominant_hgs"
    # And assign Right hand HGS value:
    df_ses2.loc[sub_id, "dominant_hgs"] = \
        df_ses2.loc[sub_id, "1_post_right_hgs"]
    df_ses2.loc[sub_id, "nondominant_hgs"] = \
        df_ses2.loc[sub_id, "1_post_left_hgs"]
    # ------------------------------------
    # If handedness is equal to 2
    # Left hand is Dominantsession
    # Find handedness equal to 2:
    sub_id = df_ses2[df_ses2['1707-0.0']== 2.0].index.values
    # Add and new column "dominant_hgs"
    # And assign Left hand HGS value:
    df_ses2.loc[sub_id, "dominant_hgs"] = \
        df_ses2.loc[sub_id, "1_post_left_hgs"]
    df_ses2.loc[sub_id, "nondominant_hgs"] = \
        df_ses2.loc[sub_id, "1_post_right_hgs"]
    # ------------------------------------
    # If handedness is equal to:
    # 3 (Use both right and left hands equally) OR
    # -3 (handiness is not available/Prefer not to answer) OR
    # NaN value
    # Dominant will be the Highest Handgrip score from both hands.
    # Find handedness equal to 3, -3 or NaN:
    sub_id = df_ses2[(df_ses2['1707-0.0']== 3.0) | (df_ses2['1707-0.0']== -3.0) | (df_ses2['1707-0.0'].isna())].index.values
    # Add and new column "dominant_hgs"
    # And assign Highest HGS value among Right and Left HGS:        
    df_ses2.loc[sub_id, f"dominant_hgs"] = df_ses2.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].max(axis=1)
    df_ses2.loc[sub_id, f"nondominant_hgs"] = df_ses2.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].min(axis=1)

    ###############################################################################
    ##############################################################################
    sub_id = df_ses3[df_ses3['1707-0.0']== 1.0].index.values
    # Add and new column "dominant_hgs"
    # And assign Right hand HGS value:
    df_ses3.loc[sub_id, "dominant_hgs"] = \
        df_ses3.loc[sub_id, "1_post_right_hgs"]
    df_ses3.loc[sub_id, "nondominant_hgs"] = \
        df_ses3.loc[sub_id, "1_post_left_hgs"]
    # ------------------------------------
    # If handedness is equal to 2
    # Left hand is Dominantsession
    # Find handedness equal to 2:
    sub_id = df_ses3[df_ses3['1707-0.0']== 2.0].index.values
    # Add and new column "dominant_hgs"
    # And assign Left hand HGS value:
    df_ses3.loc[sub_id, "dominant_hgs"] = \
        df_ses3.loc[sub_id, "1_post_left_hgs"]
    df_ses3.loc[sub_id, "nondominant_hgs"] = \
        df_ses3.loc[sub_id, "1_post_right_hgs"]
    # ------------------------------------
    # If handedness is equal to:
    # 3 (Use both right and left hands equally) OR
    # -3 (handiness is not available/Prefer not to answer) OR
    # NaN value
    # Dominant will be the Highest Handgrip score from both hands.
    # Find handedness equal to 3, -3 or NaN:
    sub_id = df_ses3[(df_ses3['1707-0.0']== 3.0) | (df_ses3['1707-0.0']== -3.0) | (df_ses3['1707-0.0'].isna())].index.values
    # Add and new column "dominant_hgs"
    # And assign Highest HGS value among Right and Left HGS:        
    df_ses3.loc[sub_id, f"dominant_hgs"] = df_ses3.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].max(axis=1)
    df_ses3.loc[sub_id, f"nondominant_hgs"] = df_ses3.loc[sub_id, [f"1_post_left_hgs", f"1_post_right_hgs"]].min(axis=1)
    ##############################################################################
    df_ses2.loc[:, f"hgs_L+R"] = \
                df_ses2.loc[:, f"46-2.0"] + df_ses2.loc[:, f"47-2.0"]

    df_ses3.loc[:, f"hgs_L+R"] = \
                df_ses3.loc[:, f"46-3.0"] + df_ses3.loc[:, f"47-3.0"]

    df_ses2.loc[:, f"hgs_left"] = \
                df_ses2.loc[:, f"46-2.0"]

    df_ses3.loc[:, f"hgs_left"] = \
                df_ses3.loc[:, f"46-3.0"]
                
    df_ses2.loc[:, f"hgs_right"] = \
                df_ses2.loc[:, f"47-2.0"]

    df_ses3.loc[:, f"hgs_right"] = \
                df_ses3.loc[:, f"47-3.0"]

    df_post = pd.concat([df_ses2, df_ses3], axis=0)
    df_post = df_post[df_post.loc[:, f"dominant_hgs"] >=4]

    ##############################################################################
    # extract_features = ExtractFeatures(df_tmp_mri, motor, population)
    # extracted_data = extract_features.extract_features()
    # Remove columns that all values are NaN
    nan_cols = df_post.columns[df_post.isna().all()].tolist()
    df_test_set = df_post.drop(nan_cols, axis=1)

    mri_features = df_test_set.copy()
        
    # X = define_features(feature_type, new_data)
    X = ["post_age", "post_bmi", "post_height", "post_waist_hip_ratio"]
    y = target

    ###############################################################################
    # Remove Missing data from Features and Target
    mri_features = mri_features.dropna(subset=y)
    mri_features = mri_features.dropna(subset=X)


    # new_data = mri_features[X]
    # new_data = new_data.rename(columns={'post_age': 'Age', 'post_bmi': '21001', 'post_height':'50', 'post_waist_hip_ratio': 'waist_to_hip_ratio'})
    # new_data = pd.concat([new_data, mri_features[y],mri_features['31-0.0']], axis=1)

    mri_features = mri_features.rename(columns={'post_age': 'Age', 'post_bmi': '21001', 'post_height':'50', 'post_waist_hip_ratio': 'waist_to_hip_ratio'})
    
    df_test_female = mri_features[mri_features['31-0.0']==0]
    df_test_male = mri_features[mri_features['31-0.0']==1]

    X = ['21001', '50', 'waist_to_hip_ratio', 'Age']

    return mri_features, df_test_female, df_test_male, X, y
