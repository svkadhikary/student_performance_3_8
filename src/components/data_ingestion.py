import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, path_to_data):
        logging.info("Entered the data ingestion method")

        try:
            df = pd.read_csv(path_to_data)
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Splitting into train and test split")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=66)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Error occured")
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion("notebook/data/stud.csv")

    data_transformation=DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()

    name, best_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print("Best score: ", best_score, "Name of the model: ", name)
