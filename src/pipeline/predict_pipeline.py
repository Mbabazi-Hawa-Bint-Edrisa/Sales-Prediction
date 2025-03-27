import numpy as np
import pandas as pd
import joblib

class CustomData:
    def __init__(self, Item_Weight, Item_Fat_Content, Item_Type, Item_MRP, Outlet_Establishment_Year, Outlet_Location, Outlet_Type):
        self.Item_Weight = float(Item_Weight)
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Type = Item_Type
        self.Item_MRP = float(Item_MRP)
        self.Outlet_Establishment_Year = float(Outlet_Establishment_Year)
        self.Outlet_Age = 2025 - self.Outlet_Establishment_Year
        self.Outlet_Location = Outlet_Location
        self.Outlet_Type = 'Supermarket' if Outlet_Type == 'Supermarket Type1' else Outlet_Type

    def get_data_as_data_frame(self):
        data = {
            'Item_Weight': [self.Item_Weight],
            'Item_Fat_Content': [self.Item_Fat_Content],
            'Item_Type': [self.Item_Type],
            'Item_MRP': [self.Item_MRP],
            'Outlet_Age': [self.Outlet_Age],
            'Outlet_Location': [self.Outlet_Location],
            'Outlet_Type': [self.Outlet_Type]
        }
        return pd.DataFrame(data)

class PredictPipeline:
    def __init__(self):
        self.pipeline = joblib.load(r'C:\Users\USER\Desktop\sales pred\artifacts\model.pkl')

    def predict(self, df):
        # Add Is_High_Fat
        df['Is_High_Fat'] = df['Item_Fat_Content'].apply(lambda x: 1 if x == 'Regular' else 0)
        
        # Log transform Item_MRP
        df['Item_MRP_log'] = np.log1p(df['Item_MRP'])

        # Reorder columns to match training
        df = df[['Item_Weight', 'Item_MRP_log', 'Outlet_Age', 'Is_High_Fat', 'Item_Fat_Content', 'Item_Type', 'Outlet_Location', 'Outlet_Type']]

        # Predict using the pipeline (handles preprocessing and prediction)
        pred_log = self.pipeline.predict(df)[0]
        pred_ugx = np.expm1(pred_log)  # Convert back to UGX
        return [pred_ugx]