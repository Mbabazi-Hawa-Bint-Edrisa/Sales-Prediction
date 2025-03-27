import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            # Define categorical and numerical columns
            cat_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location', 'Outlet_Type']
            num_features = ['Item_Weight', 'Item_MRP_log', 'Outlet_Age', 'Is_High_Fat']

            # Pipeline for numerical features: impute with mean, then scale
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ('scaler', StandardScaler())
            ])

            # Pipeline for categorical features: impute with most frequent, then one-hot encode
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder(drop='first', sparse_output=False)),
                ('scaler', StandardScaler())
            ])

            logging.info(f"Features to one-hot encode: {cat_features}")
            logging.info(f"Features to scale: {num_features}")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, num_features),
                    ('cat', cat_pipeline, cat_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def inititate_data_transformation(self):
        try:
            # Load the cleaned datasets
            train_df = pd.read_csv("artifacts/train_cleaned.csv")
            test_df = pd.read_csv("artifacts/test_cleaned.csv")         

            logging.info("Read train and test data completed")
            logging.info("Train dataset columns: %s", train_df.columns.tolist())
            logging.info("Test dataset columns: %s", test_df.columns.tolist())

            # Add derived features to match predict_pipeline.py
            train_df['Is_High_Fat'] = train_df['Item_Fat_Content'].apply(lambda x: 1 if x == 'Regular' else 0)
            test_df['Is_High_Fat'] = test_df['Item_Fat_Content'].apply(lambda x: 1 if x == 'Regular' else 0)

            train_df['Item_MRP_log'] = np.log1p(train_df['Item_MRP'])
            test_df['Item_MRP_log'] = np.log1p(test_df['Item_MRP'])

            logging.info("Added derived features: Is_High_Fat, Item_MRP_log")

            # Log-transform the target
            target_column_name = "Item_Outlet_Sales"
            train_df['Item_Outlet_Sales_log'] = np.log1p(train_df[target_column_name])
            test_df['Item_Outlet_Sales_log'] = np.log1p(test_df[target_column_name])

            logging.info("Log-transformed target: Item_Outlet_Sales_log")

            # Prepare input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name, 'Item_Outlet_Sales_log', 'Item_MRP'], axis=1)
            target_feature_train_df = train_df['Item_Outlet_Sales_log']
            input_feature_test_df = test_df.drop(columns=[target_column_name, 'Item_Outlet_Sales_log', 'Item_MRP'], axis=1)
            target_feature_test_df = test_df['Item_Outlet_Sales_log']

            logging.info("Input features for training: %s", input_feature_train_df.columns.tolist())

            # Get the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation()

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object
            logging.info("Saving preprocessing object")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_file = data_transformation.inititate_data_transformation()