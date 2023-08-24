
from hgsprediction.data_preprocessing import HealthyDataPreprocessor

def run_healthy_preprocessing(df, session):
    
    data_processor = HealthyDataPreprocessor(df, session)

    # Call functions inside the class
    # CHECK HGS AVAILABILITY
    data = data_processor.check_hgs_availability(df)
    # DATA VALIDATION
    data = data_processor.validate_handgrips(data)

    data = data_processor.remove_nan_columns(data)

    return data