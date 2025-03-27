import sys
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataCleaningConfig:
    train_data_path_cleaned: str = os.path.join('artifacts', "train_cleaned.csv")
    test_data_path_cleaned: str = os.path.join('artifacts', "test_cleaned.csv")

class DataCleaning:
    def __init__(self):
        self.cleaning_config = DataCleaningConfig()
    
    def initiate_data_cleaning(self):
        logging.info("Entered the data cleaning method or component")
        try:
            df_train = pd.read_csv("artifacts/train.csv")
            df_test = pd.read_csv("artifacts/test.csv")
        
            logging.info('Read the dataset as dataframe')

            # Log missing values for debugging
            logging.info("Missing values in train dataset:\n%s", df_train.isna().sum())
            logging.info("Missing values in test dataset:\n%s", df_test.isna().sum())

            # Create new column: Outlet_Age
            df_train['Outlet_Age'] = df_train['Outlet_Establishment_Year'].apply(lambda year: 2025 - year).astype(int)
            df_test['Outlet_Age'] = df_test['Outlet_Establishment_Year'].apply(lambda year: 2025 - year).astype(int)

            # Standardize values in the 'Item_Fat_Content' column
            df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
            df_test['Item_Fat_Content'] = df_test['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

            # Drop unnecessary columns
            columns_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Visibility', 'Outlet_Establishment_Year']
            df_train.drop(columns=columns_to_drop, axis=1, inplace=True)
            df_test.drop(columns=columns_to_drop, axis=1, inplace=True)

            # Log the remaining columns
            logging.info("Columns in cleaned train dataset: %s", df_train.columns.tolist())
            logging.info("Columns in cleaned test dataset: %s", df_test.columns.tolist())

            # Save the cleaned datasets
            os.makedirs(os.path.dirname(self.cleaning_config.train_data_path_cleaned), exist_ok=True)
            logging.info('Directory created')

            df_train.to_csv(self.cleaning_config.train_data_path_cleaned, index=False, header=True)
            logging.info('Train data saved')
            df_test.to_csv(self.cleaning_config.test_data_path_cleaned, index=False, header=True)
            logging.info('Test data saved')

            logging.info("Train and test data cleaned")
            return df_train, df_test
            
        except Exception as e:
            raise CustomException(e, sys)