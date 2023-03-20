import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_columns = ['writing_score', 'reading_score']
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("oh_encoder", OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and categorical columns scaling and encoding complete")

            preprocessor = ColumnTransformer(
                [
                ("num_transformer", num_pipeline, num_columns),
                ("cat_transformer", cat_pipeline, cat_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info("Error occured")
            raise CustomException(str(e), sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data in dataframe")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_col_name = "math_score"

            input_features_train = train_df.drop(target_col_name, axis=1)
            target_feature_train = train_df[target_col_name]

            input_features_test = test_df.drop(target_col_name, axis=1)
            target_feature_test = test_df[target_col_name]

            logging.info("Applying preprocessing train and test data")

            input_processed_train = preprocessor_obj.fit_transform(input_features_train)
            input_processed_test = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_processed_train, np.array(target_feature_train)]
            test_arr = np.c_[input_processed_test, np.array(target_feature_test)]

            save_object(
                obj = preprocessor_obj,
                file_path = self.data_transformation_config.preprocessor_file_path
            )

            logging.info("Data Transformation complete")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )


        except Exception as e:
            logging.info("Error Occured")
            raise CustomException(e, sys)

