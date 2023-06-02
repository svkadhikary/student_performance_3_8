# Student Performance Project

The Student Performance Project is a web-based application that analyzes student performance data, provides various machine learning functionalities, and exposes a Flask API for easy interaction.

## Features

- Data preprocessing: Clean the dataset, handle missing values, encode categorical variables, and normalize numerical features.
- Exploratory data analysis: Visualize and analyze the relationships between different variables in the dataset.
- Model training and evaluation: Train machine learning models using different algorithms, evaluate their performance, and select the best model.
- Predictive analysis: Use the trained models to make predictions on new student data.
- Flask API: Expose endpoints for uploading datasets, training models, saving trained models, and making predictions.
- Web Interface: Interact with the project functionalities through a user-friendly web interface.

## Dataset

The project uses a dataset of student performance, which includes various attributes such as demographic information, education-related factors, and exam scores. The dataset can be uploaded through the web interface or via the Flask API endpoints.

## Setup and Installation

To run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/svkadhikary/student_performance_3_8.git`
2. Navigate to the project directory: `cd student_performance_3_8`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Flask application: `python application.py`
5. Access the web interface in your browser at `http://localhost:5000` to interact with the project.

## Dependencies

The project requires the following Python libraries:

- flask
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


## Web Interface

The web interface allows you to perform the following actions:

- Upload a dataset: Select and upload a student performance dataset in CSV format.
- Train models: Choose different models and their parameters, and initiate the training process.
- Save models: Save the trained models for future use.
- Predictions: Make predictions on individual student records or entire datasets using the trained models.

## Tested Environment

This project has been tested and verified to work on the following environment:

- Microsoft Azure Web App

## Acknowledgments

- Thanks to [@krishnaik06](https://github.com/krishnaik06).

- The code in this repository is inspired by various ML implementations and tutorials available in the Machine Learning Community.

## Contribution

Contributions to the project are welcome! If you find any issues, have suggestions, or want to add new features, please feel free to create a pull request.


