from flask import Flask, request, render_template, redirect, flash, url_for, session
import pandas as pd
import os
import yaml

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from src.pipeline.train_pipeline import DataTrainingPipeline


application = Flask(__name__)
app = application

app.config['UPLOAD_FOLDER'] = './uploads'
app.secret_key = "super_secret_key"

@app.route("/")
def index():
    return redirect(url_for('upload_file'))

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
        <body text-align: center>
            <form method=post enctype=multipart/form-data>
                <input type=file name=file>
                <input type=submit value=Upload>
            </form>
        </body>
    </html>
    '''


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


@app.route("/model_selection", methods=['POST'])
def model_selection():
    filename = session.get('filepath')
    
    if request.method == 'POST':
        flash("Training model now")
        # fetching models and parameters from the html form
        selected_models = {}
        for model in session.get('models'):
            if request.form.get(model):
                selected_models[model] = {}
                for param in request.form.getlist(model + '[]'):
                    param_name, param_val = param.split("|")
                    selected_models[model][param_name] = param_val

        train_pipe = DataTrainingPipeline(filename)
        name, best_score = train_pipe.train_model(selected_models)
        
        return render_template("success.html", model_name=name, model_score=best_score)

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template("predict.html")
    
    elif request.method == 'POST':
        
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

        pred_pipeline = PredictionPipeline()
        prediction = pred_pipeline.predict(df_pred)

        return render_template("predict.html", results=prediction[0])
    

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)

