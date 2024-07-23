
import sys
import os
import pandas as pd
import numpy as np

#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set = sys.argv[9]
gender = sys.argv[10]   

###############################################################################
for session in ["0", "1"]:
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "results_hgsprediction",
        "test_retest_reliability",
        f"{population}",
        f"{mri_status}",
        f"{feature_type}",
        f"{target}",
        f"longitudinal_session_ukb",
        f"{session}_session_ukb",
        "extracted_data_by_feature_and_target",
        )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
    folder_path,
    f"{gender}_extracted_data_by_feature_and_target.csv")

    # Save the dataframe to csv file path
    if session == "0":
        df_session_0 = pd.read_csv(file_path, sep=',', index_col=0)
    elif session == "1":
        # Save the dataframe to csv file path
        df_session_1 = pd.read_csv(file_path, sep=',', index_col=0)


###############################################################################
# Function to calculate the Concordance correlation coefficient (CCC):
# CCC = 2⋅Cov(X,Y)/​Var(X)+Var(Y)+(Xˉ−Yˉ)^2
def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    ccc = numerator / denominator
    
    return ccc

###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
y_true_session_0 = df_session_0[f"{target}"]
y_pred_session_0 = df_session_0[f"{target}_corrected_predicted"]

ccc_value = concordance_correlation_coefficient(y_true_session_0, y_pred_session_0)
print(f"Concordance Correlation Coefficient: {ccc_value:.2f}")
###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 1

y_true_session_1 = df_session_1[f"{target}"]
y_pred_session_1 = df_session_1[f"{target}_corrected_predicted"]

ccc_value = concordance_correlation_coefficient(y_true_session_1, y_pred_session_1)
print(f"Concordance Correlation Coefficient: {ccc_value:.2f}")
###############################################################################





print("===== Done! =====")
embed(globals(), locals())