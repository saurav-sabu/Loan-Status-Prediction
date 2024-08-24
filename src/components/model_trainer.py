import os
import sys
from dataclasses import dataclass

from src.utils import evaluate_model, save_obj
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):

        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "DecisionTree":DecisionTreeClassifier(),
                "SVC":SVC(),
                "LogisticRegression":LogisticRegression(),
                "RandomForest":RandomForestClassifier(),
                "AdaBoost":AdaBoostClassifier(),
                "GradientBoost":GradientBoostingClassifier(),
                "XGBoost":XGBClassifier(),
                "KNN":KNeighborsClassifier()
            }

            model_report = evaluate_model(X_train,y_train,X_test,y_test,models)

            best_model_accuracy = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_accuracy)  
            ]

            best_model =  models[best_model_name]

            if best_model_accuracy<0.6:
                raise CustomException("No best model found")
            
            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            pred = best_model.predict(X_test)

            return accuracy_score(y_test,pred)

        except Exception as e:
            raise CustomException(e,sys)
