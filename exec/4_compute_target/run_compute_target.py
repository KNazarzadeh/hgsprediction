from hgsprediction.data_preprocessing import healthy_data_preprocessor, DataPreprocessor

def run_healthy_preprocessing(df, session):
    
    data_processor = DataPreprocessor(df, session)

    # Call all functions inside the class
    # Calculate target
    data = data_processor.sum_handgrips(data)
    data = data_processor.calculate_left_hgs(data)
    data = data_processor.calculate_right_hgs(data)
    data = data_processor.sub_handgrips(data)
    data = data_processor.calculate_laterality_index(data)
    data = data_processor.remove_nan_columns(data)

    return data