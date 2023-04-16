import os
import sys
from datetime import datetime

from src.utils import save_object, load_object
from src.exception import CustomException


class ModelManager:
    def __init__(self):
        self.models = []
        self.model_path = "artifacts/models"

    def add_model(self, model, name, score):
        self.models.append((model, name, score))

    def remove_model(self, name):
        self.models = [(model, m_name, m_score) for model, m_name, m_score in self.models if m_name != name]

    def get_models(self):
        return self.models

    def check_model_existance(self, model_name, model_score):
        try:
            models = [f for f in os.listdir(self.model_path) if f.endswith('.pkl')]
            print(models)

            for model in models:
                if model_name == model.split("_")[0]:
                    print(model, " found")
                    saved_model_score = model.split(".pkl")[0].split("_")[-1]
                    if float(saved_model_score) >= float(model_score):
                        print(saved_model_score, " is higher")
                        return True
                    else:
                        print("saved model score is lower, deleting model")
                        os.remove(os.path.join(self.model_path, model))
                    return False
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_model(self, models_to_save, model_report):
        try:
            for model_name in models_to_save:
                print("checking ", model_name)
                score = model_report[model_name]['score']
                ## check if the saved model has better score
                if self.check_model_existance(model_name, score):
                    continue
                if model_name in model_report:
                    # get the score and the model from the dict
                    model = model_report[model_name]['model']
                    # create save model name with model name, model score, and datetime
                    model_name = model_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "_" + str(score) + ".pkl"
                    save_path = os.path.join(self.model_path, model_name)
                    # save the model
                    save_object(model, save_path)
            return True
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def load_model(self, filepath):
        try:
            return load_object(file_path=filepath)
        except Exception as e:
            raise CustomException(e, sys)


