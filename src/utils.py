import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def save_obj(file_path,obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Training set performance
            model_train_accuracy = accuracy_score(y_train, y_train_pred)
            model_train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
            model_train_classification_report = classification_report(y_train, y_train_pred)

            # Test set performance
            model_test_accuracy = accuracy_score(y_test, y_test_pred)
            model_test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
            model_test_classification_report = classification_report(y_test, y_test_pred)

            report[list(models.keys())[i]] = model_test_accuracy

        return report 
    
    except Exception as e:
        raise CustomException(e,sys)