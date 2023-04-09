import os
import sys
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class DataTrainingPipeline:
    def __init__(self, filepath) -> None:
        self.filepath = filepath

    def data_preprocessing(self):
        try:
            logging.info("Data ingestion started")
            data_ingestion_obj = DataIngestion()
            train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion(self.filepath)
            logging.info("Data transformation started")
            data_transformation_obj = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)
    
    def train_model(self, selected_model_w_param):
        try:
            logging.info("Model training started")
            model_trainer_obj = ModelTrainer()
            train_arr, test_arr = self.data_preprocessing()
            name, best_score = model_trainer_obj.initiate_model_trainer(train_arr, test_arr, selected_model_w_param)
            logging.info(f"Best model found {name} with score {best_score}")
            return (name, best_score)

        except Exception as e:
            raise CustomException(e, sys)