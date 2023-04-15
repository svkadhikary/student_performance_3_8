import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        self.models_path = "artifacts/models/"
        self.preprocessor_path = 'artifacts/preprocessor.pkl'

    def get_saved_models(self):
        try:
            saved_models = {}
            for model in os.listdir(self.models_path):
                if model.endswith(".pkl"):
                    model_name = model.split("_")[0]
                    model_score = model.split(".pkl")[0].split("_")[-1]
                    saved_models[model_name] = model_score
            return saved_models
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features, model):
        try:
            select_model = ""
            for model_name in os.listdir(self.models_path):
                if model in model_name:
                    select_model = os.path.join(self.models_path, model_name)

            model = load_object(file_path=select_model)
            preprocessor = load_object(file_path=self.preprocessor_path)

            data_transformed = preprocessor.transform(features)

            preds = model.predict(data_transformed)

            return preds
        except Exception as e:
            raise CustomException(e, sys)




class CustomData:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int) -> None:
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    
    def get_data_as_dataframe(self):
        try:    
            custom_data_input_dict = {
                    "gender": [self.gender],
                    "race_ethnicity": [self.race_ethnicity],
                    "parental_level_of_education": [self.parental_level_of_education],
                    "lunch": [self.lunch],
                    "test_preparation_course": [self.test_preparation_course],
                    "reading_score": [self.reading_score],
                    "writing_score": [self.writing_score],
                }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
