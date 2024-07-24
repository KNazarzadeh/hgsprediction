
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
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
def concordance_correlation_coefficient(test_session_0, retest_session_1):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'test_session_0': test_session_0,
        'retest_session_1': retest_session_1
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    test_session_0 = df['test_session_0']
    retest_session_1 = df['retest_session_1']
    cor = np.corrcoef(test_session_0, retest_session_1)[0][1]
    # Means
    mean_true = np.mean(test_session_0)
    mean_pred = np.mean(retest_session_1)
    # Population variances
    var_true = np.var(test_session_0)
    var_pred = np.var(retest_session_1)
    # Population standard deviations
    sd_true = np.std(test_session_0)
    sd_pred = np.std(retest_session_1)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    ccc = numerator / denominator
    
    return ccc

###############################################################################
print("\ngender: ", gender)
print("----------------------------------------------------------")
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_predicted"]
retest_session_1 = df_session_1[f"{target}_predicted"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"Concordance Correlation Coefficient for predicted values without bias-adjustment: {ccc_value:.2f}")
###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_corrected_predicted"]
retest_session_1 = df_session_1[f"{target}_corrected_predicted"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"Concordance Correlation Coefficient for predicted values with bias-adjustment: {ccc_value:.2f}")
###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_delta(true-predicted)"]
retest_session_1 = df_session_1[f"{target}_delta(true-predicted)"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"\nConcordance Correlation Coefficient for delta values without bias-adjustment: {ccc_value:.2f}")
###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_corrected_delta(true-predicted)"]
retest_session_1 = df_session_1[f"{target}_corrected_delta(true-predicted)"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"Concordance Correlation Coefficient for delta values with bias-adjustment: {ccc_value:.2f}")
###############################################################################
age_differ = df_session_1.loc[:, "age"] - df_session_0.loc[:, "age"]
min_age = min(age_differ)
max_age = max(age_differ)

###############################################################################
print("----------------------------------------------------------")

y_true_session_0 = df_session_0[f"{target}"]
y_pred_session_0 = df_session_0[f"{target}_predicted"]
mae_ses_0_predicted = mean_absolute_error(y_true_session_0, y_pred_session_0)
print(f"MAE for predicted values of session 0 without bias-adjustment: {mae_ses_0_predicted:.3f}")

y_true_session_1 = df_session_1[f"{target}"]
y_pred_session_1 = df_session_1[f"{target}_predicted"]
mae_ses_1_predicted = mean_absolute_error(y_true_session_1, y_pred_session_1)
print(f"MAE for predicted values of session 1 without bias-adjustment: {mae_ses_1_predicted:.3f}")

y_true_session_0 = df_session_0[f"{target}"]
y_pred_session_0 = df_session_0[f"{target}_corrected_predicted"]
mae_ses_0_corrected_predicted = mean_absolute_error(y_true_session_0, y_pred_session_0)
print(f"\nMAE for predicted values of session 0 with bias-adjustment: {mae_ses_0_corrected_predicted:.3f}")

y_true_session_1 = df_session_1[f"{target}"]
y_pred_session_1 = df_session_1[f"{target}_corrected_predicted"]
mae_ses_1_corrected_predicted = mean_absolute_error(y_true_session_1, y_pred_session_1)
print(f"MAE for predicted values of session 1 with bias-adjustment: {mae_ses_1_corrected_predicted:.3f}")

print("===== Done! =====")
embed(globals(), locals())