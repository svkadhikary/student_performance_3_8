import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
                )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            hyperparams = {
                "Random Forest": {
                    "n_estimators": [100, 500, 1000],
                    "max_features": ["sqrt", "log2"],
                    "max_depth": [5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Decision Tree": {
                    'max_depth': [3, 5, 7, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 500, 1000],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'subsample': [0.5, 0.8, 1.0],
                    'loss': ['ls', 'lad', 'huber', 'quantile']
                },
                "Linear Regression": {
                    'fit_intercept': [True, False]
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [5, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2],
                },
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'n_estimators': [50, 100, 200],
                    'reg_alpha': [0.1, 1, 10],
                    'reg_lambda': [0.1, 1, 10]
                },
                "CatBoosting Classifier": {
                    'iterations': [100, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5],
                    'loss_function': ['MAE', 'RMSE']
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                    'loss': ['linear', 'square', 'exponential']
                }
            }
            
            # evaluate which model gives best score
            model_report, best_model = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, hyperparams=hyperparams)
            
            model_report = {k: v for k, v in sorted(model_report.items(), key=lambda item: item[1], reverse=True)}

            best_model_name, best_model_score = next(iter(model_report.items()))

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")
            print("Best model:", best_model)

            save_object(
                obj = best_model,
                file_path = self.model_trainer_config.trained_model_file_path
            )
            logging.info("Best model saved")

            return (best_model_name, best_model_score)

        except Exception as e:
            raise CustomException(e, sys)
        

