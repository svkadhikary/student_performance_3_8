from flask import Flask, request, render_template, redirect, flash, url_for, session, Response
import pandas as pd
import os
import yaml

from src.components.model_manager import ModelManager
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from src.pipeline.train_pipeline import DataTrainingPipeline

# initialize flask app
application = Flask(__name__)
app = application
# setup upload folder for dataset uploads
app.config['UPLOAD_FOLDER'] = './uploads'
# configure secret key
app.secret_key = "super_secret_key"

# configure global variable
model_report = {}

# index page; todo: other modifications
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        inp = request.form.get('inp')
        if inp == "upload":
            return redirect(url_for('upload_file'))
        return redirect(url_for('predict_data'))
    return '''
    <!doctype html>
    <html>
    <body style="text-align: center;">
    <h2>Upload dataset and train different models or predict values based on saved trained models</h2>
        <div class="container">
            <form method=post>
                <input type='submit' value=Predict name=inp>
                <div class="row"></div><br>
                <input type='submit' value=upload name=inp>
            </form>
        </div>
    </body>
    </html>
    '''

# dataset upload route
@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            # Save the file to the app folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            session['filepath'] = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            return redirect(url_for('train'))
    
    return'''
    <!doctype html>
    <html>
        <body style="text-align: center;">
            <h3>Upload the dataest to train with</h3>
            <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
            <br>
        </body>
    </html>
    '''

# dataset review and model selection
@app.route('/train', methods=['GET', 'POST'])
def train():
    filename = session.get('filepath')
    df = pd.read_csv(filename)
    if request.method == 'POST':
        with open('models.yaml', 'r') as f:
            models = yaml.safe_load(f)
            session['models'] = list(models.keys())
        return render_template('model_selection.html', models=models)
    # Render training page with button to go to model selection page
    return render_template('train.html', data=df.to_html(index=False))

# Model and hyperparameter selection page
@app.route("/model_selection", methods=['POST'])
def model_selection():
    filename = session.get('filepath')
    
    if request.method == 'POST':
        # fetching models and parameters from the html form
        selected_models = {}
        for model in session.get('models'):
            if request.form.get(model):
                selected_models[model] = {}
                for param in request.form.getlist(model + '[]'):
                    param_name, param_val = param.split("|")
                    selected_models[model][param_name] = param_val

        train_pipe = DataTrainingPipeline(filename)
        global model_report
        best_model, best_score, model_report = train_pipe.train_model(selected_models)
        
        models = {model_name: model_report[model_name]['score'] for model_name in model_report}
        
        return render_template("success.html", model_name=best_model, model_score=best_score, models=models)

# Save user-selected models
@app.route('/save_models', methods=['POST'])
def save_models():
    if request.method == 'POST':
        # get selected models from form
        models = request.form.getlist('models_to_save')
        models = [val.split("-")[0] for val in models]
        global model_report
        # save selected models
        model_manager = ModelManager()
        res = model_manager.save_model(models, model_report)

        if res:
            flash("Models saved")
            return redirect(url_for('predict_data'))

        return Response("Error")

# Single row data prediction
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_data():
    
    pred_pipeline = PredictionPipeline()
    saved_models = pred_pipeline.get_saved_models()

    if request.method == 'GET':
        return render_template("predict.html", models=saved_models)
    
    elif request.method == 'POST':
        
        model = request.form['model']
        print(model)

        # Get the selected prediction mode from the radio button
        prediction_mode = request.form.get('prediction_mode')

        if prediction_mode == 'single_row':
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            df_pred = data.get_data_as_dataframe()
            print(df_pred)

            prediction = pred_pipeline.predict(df_pred, model)

            return render_template("predict.html", results=prediction[0], models=saved_models)
        
        elif prediction_mode == 'entire_dataset':
            file = request.files['dataset']
            if file:
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                prediction = pred_pipeline.predict_on_batch(filepath, model)

            return render_template("predict_on_batch.html", df=prediction.to_html(index=False))


## todo: evaulation

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)

