


###############################################################################     
def save_scores_trained(
    df,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    samplesize,
):
    if confound_status == "0":
        confound = "without_confound_removal"
    elif confound_status == "1":
        confound = "with_confound_removal"

    # Assuming that you have already trained and instantiated the model as `model`
    folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "project_hgsprediction",
            "results_hgsprediction",
            f"{population}",
            f"{mri_status}",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"{gender}",
            "multi_samplesize_results",
            f"results_samples_{samplesize}",
            "scores_trained",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    