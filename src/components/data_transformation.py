import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import save_obj

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_transformed_data(self):
        try:
            logging.info("")
            
            categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Property_Area']
            numerical_features = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OrdinalEncoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Categorical & Numerical Pipeline Created")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_transformed_data()

            target_column_name  = "Loan_Status"

            train_df[target_column_name] = train_df[target_column_name].map({'Y': 1, 'N': 0})
            test_df[target_column_name] = test_df[target_column_name].map({'Y': 1, 'N': 0})

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            save_obj(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                     obj = preprocessor_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)