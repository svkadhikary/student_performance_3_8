import os
import sys
from dataclasses import dataclass
import importlib
import yaml

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    filename = f"best_model.pkl"
    trained_model_file_path = os.path.join("artifacts/models", filename)


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    

    # fetch the module and classes from the models.yaml
    def get_model_class_name(self, model_name):
        try:
            with open("models.yaml", "r") as yaml_file:
                model_conf = yaml.safe_load(yaml_file)
                if model_name in model_conf.keys():
                    model_module = model_conf[model_name]['module']
                    model_class = model_conf[model_name]['class']
                return (model_module, model_class)
        except Exception as e:
            raise CustomException(e, sys)
        

    # importing selected models
    def model_imports(self, selected_models):
        try:
            model_classes = {}
            for model_name in selected_models:
                model_module_name, model_class_name = self.get_model_class_name(model_name)
                model_module = importlib.import_module(model_module_name)
                model_class = getattr(model_module, model_class_name)
                model_classes[model_name] = model_class
            
            return model_classes
        except Exception as e:
            raise CustomException(e, sys)
        
    def model_param_conf(self, selected_models_w_params):
        try:
            with open("models.yaml", 'rb') as yaml_file:
                models = yaml.safe_load(yaml_file)
            
            model_params = {}
            for model_name, model_params_dict in selected_models_w_params.items():
                model_params[model_name] = {}
                for param_name, param_values in model_params_dict.items():
                    
                    param_values_yaml = models[model_name]['params'][param_name]
                    model_params[model_name][param_name] = param_values_yaml
            
            return model_params

        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_model_trainer(self, train_arr, test_arr, selected_models_w_param):
        try:
            logging.info("Split training and test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
                )
            
            logging.info("Importing model classes")
            models_classes = self.model_imports(selected_models_w_param)
            logging.info("configuring model parameters")
            model_params = self.model_param_conf(selected_models_w_param)

            # models = {
            #     "Random Forest": RandomForestRegressor(),
            #     "Decision Tree": DecisionTreeRegressor(),
            #     "Gradient Boosting": GradientBoostingRegressor(),
            #     "Linear Regression": LinearRegression(),
            #     "K-Neighbors Regressor": KNeighborsRegressor(),
            #     "XGBRegressor": XGBRegressor(),
            #     "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            #     "AdaBoost Regressor": AdaBoostRegressor()
            # }

            # hyperparams = {
            #     "Random Forest": {
            #         "n_estimators": [100, 500, 1000],
            #         "max_features": ["sqrt", "log2"],
            #         "max_depth": [5, 10, 15],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 4]
            #     },
            #     "Decision Tree": {
            #         'max_depth': [3, 5, 7, 15],
            #         'min_samples_split': [2, 5, 10],
            #         'min_samples_leaf': [1, 2, 4],
            #         'max_features': ['auto', 'sqrt', 'log2']
            #     },
            #     "Gradient Boosting": {
            #         'n_estimators': [100, 500, 1000],
            #         'max_depth': [3, 5, 7],
            #         'learning_rate': [0.01, 0.1, 0.5],
            #         'subsample': [0.5, 0.8, 1.0],
            #         'loss': ['ls', 'lad', 'huber', 'quantile']
            #     },
            #     "Linear Regression": {
            #         'fit_intercept': [True, False]
            #     },
            #     "K-Neighbors Regressor": {
            #         'n_neighbors': [5, 10, 15],
            #         'weights': ['uniform', 'distance'],
            #         'p': [1, 2],
            #     },
            #     "XGBRegressor": {
            #         'learning_rate': [0.01, 0.1, 0.2],
            #         'max_depth': [3, 5, 7],
            #         'n_estimators': [50, 100, 200],
            #         'reg_alpha': [0.1, 1, 10],
            #         'reg_lambda': [0.1, 1, 10]
            #     },
            #     "CatBoosting Regressor": {
            #         'iterations': [100, 500, 1000],
            #         'learning_rate': [0.01, 0.05, 0.1],
            #         'depth': [4, 6, 8],
            #         'l2_leaf_reg': [1, 3, 5],
            #         'loss_function': ['MAE', 'RMSE']
            #     },
            #     "AdaBoost Regressor": {
            #         'n_estimators': [50, 100, 200],
            #         'learning_rate': [0.01, 0.1, 1],
            #         'loss': ['linear', 'square', 'exponential']
            #     }
            # }
            
            # evaluate which model gives best score

            logging.info("Evaluating models")
            model_report, best_model = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models_classes,
                hyperparams=model_params
                )
            
            model_report = dict(sorted(model_report.items(), key=lambda x: x[1]['score'], reverse=True))

            best_model_tuple = next(iter(model_report.items()))

            best_model_name = best_model_tuple[0]
            best_model_score = best_model_tuple[1]['score']

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")
            print("Best model:", best_model)


            save_object(
                obj = best_model,
                file_path = self.model_trainer_config.trained_model_file_path
            )
            logging.info("Best model saved")

            del models_classes

            return (best_model_name, best_model_score, model_report)

        except Exception as e:
            raise CustomException(e, sys)
        

