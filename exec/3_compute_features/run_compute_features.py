from hgsprediction.data_preprocessing import healthy_data_preprocessor, DataPreprocessor

def run_healthy_preprocessing(df, session):
    
    data_processor = DataPreprocessor(df, session)

    # Call all functions inside the class
    # FEATURE ENGINEERING
    data = data_processor.calculate_qualification(data)
    data = data_processor.calculate_waist_to_hip_ratio(data)
    data = data_processor.calculate_neuroticism_score(data)
    data = data_processor.calculate_anxiety_score(data)
    data = data_processor.calculate_depression_score(data)
    data = data_processor.calculate_cidi_score(data)
    data = data_processor.preprocess_behaviours(data)

    data = data_processor.remove_nan_columns(data)

    return data