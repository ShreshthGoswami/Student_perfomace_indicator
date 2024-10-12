import os
import sys
import numpy as np 
import pandas as pd
import pickle  # or use dill if needed
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save the object to a file using pickle (or dill).
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified path
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # or use dill.dump if you're using dill

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate different models using GridSearchCV and return a performance report.
    """
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            # Perform grid search with cross-validation
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            # Use the best found model
            best_model = gs.best_estimator_

            # Predictions and scoring
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate R2 score for training and testing datasets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R2 score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load a pickled object from the given file path.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  # or use dill.load if you're using dill

    except Exception as e:
        raise CustomException(e, sys)
