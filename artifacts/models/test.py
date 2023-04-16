import os

def check_model_existance(model_name, model_score):
    models = [f for f in os.listdir("./") if f.endswith('.pkl')]
    # print(models)

    for model in models:
        if model_name == model.split("_")[0]:
            print(model, " found")
            saved_model_score = model.split(".pkl")[0].split("_")[-1]
            if float(saved_model_score) >= float(model_score):
                print(saved_model_score, " is higher")
                return True
            else:
                print("saved model score is lower, deleting model")
                ##os.remove(os.path.join(self.model_path, model))
                print("Removed file")
                return False
    

if __name__ == "__main__":
    for model in ['RandomForestRegressor', 'DecisionTreeRegressor', 'XGBRegressor', 'CatBoostRegressor']:
        check_model_existance(model, "0.85")