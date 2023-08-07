
from .healthy_data_preprocessor import DataPreprocessor

def run_healthy_preprocessing(data):
    
    data_processor = DataPreprocessor(data,session=0)

    # Call all functions inside the class
    # DATA VALIDATION
    data = data_processor.validate_handgrips(data)
    # FEATURE ENGINEERING
    data = data_processor.calculate_qualification(data)
    data = data_processor.calculate_waist_to_hip_ratio(data)
    data = data_processor.calculate_neuroticism_score(data)
    data = data_processor.calculate_anxiety_score(data)
    data = data_processor.calculate_depression_score(data)
    data = data_processor.calculate_cidi_score(data)
    data = data_processor.preprocess_behaviours(data)
    data = data_processor.sum_handgrips(data)
    data = data_processor.calculate_left_hgs(data)
    data = data_processor.calculate_right_hgs(data)
    data = data_processor.sub_handgrips(data)
    data = data_processor.calculate_laterality_index(data)
    data = data_processor.remove_nan_columns(data)

    return data

#     # Get the processed data
#     processed_data = data_processor.get_processed_data()

#     return processed_data

# # if __name__ == "__main__":
# #     processed_data = run_healthy_preprocessing()