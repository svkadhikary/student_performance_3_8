import os
import sys
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path:str):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models:dict, hyperparams:dict):
    try:
        report = {}
        best_model = None
        best_score = float('-inf')
        for model_name, model in models.items():
            if model_name in hyperparams:

                param_grid = hyperparams[model_name]
                grid_search = GridSearchCV(model, param_grid, scoring='r2', n_jobs=3, cv = 5 , verbose=5)
                grid_search.fit(X_train, y_train)

                current_best_model = grid_search.best_estimator_

                y_train_pred = current_best_model.predict(X_train)
                y_test_pred = current_best_model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_score

                if test_score > best_score:
                    best_score = test_score
                    best_model = current_best_model
            
            else:
                model.fit(X_train, y_train)
                test_score = r2_score(y_test, model.predict(X_test))
                report[model_name] = test_score

                if test_score > best_score:
                    best_score = test_score
                    best_model = model
            
            print("Test score: ", test_score, "for model", model_name)
                

        return (report, best_model)

    except Exception as e:
        raise CustomException(e, sys)
    
