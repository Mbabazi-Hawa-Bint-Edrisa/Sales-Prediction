from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        print("Form data received:", request.form)
        try:
            data = CustomData(
                Item_Weight=request.form.get("Item_Weight") or "0",
                Item_Fat_Content=request.form.get('Item_Fat_Content') or "Low Fat",
                Item_Type=request.form.get('Item_Type') or "Others",
                Item_MRP=request.form.get('Item_MRP') or "0",
                Outlet_Establishment_Year=request.form.get('Outlet_Establishment_Year') or "2017",  # Default to 2017
                Outlet_Location=request.form.get('Outlet_Location'),  # From form
                Outlet_Type=request.form.get('Outlet_Type') or "Supermarket Type1"  # From form
            )
        except ValueError as e:
            return render_template('home.html', results=f"Error: Invalid input- {str(e)}")

        pred_df = data.get_data_as_data_frame()
        print("Prediction DataFrame:", pred_df)

        predict_pipeline = PredictPipeline()
        try:
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=f'Predicted Sales: {results[0]:,.2f} UGX')
        except Exception as e:
            return render_template('home.html', results=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # from flask import Flask, request, render_template
# # import numpy as np
# # import pandas as pd
# # from sklearn.preprocessing import StandardScaler
# # from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# # application = Flask(__name__)
# # app = application

# # ## Route for a home page
# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/predictdata', methods=['GET', 'POST'])
# # def predict_datapoint():
# #     if request.method == 'GET':
# #         return render_template('home.html')
# #     else:
# #         data = CustomData(
# #             Item_Weight=request.form.get("Item_Weight"),
# #             Item_Fat_Content=request.form.get('Item_Fat_Content'),
# #             Item_Type=request.form.get('Item_Type'),
# #             Item_MRP=request.form.get('Item_MRP'),
# #             Outlet_Location_Type=request.form.get('Outlet_Location'),
# #             Outlet_Type=request.form.get('Outlet_Type'),
# #             Outlet_Age=request.form.get('Outlet_Age')
# #         )

# #         pred_df = data.get_data_as_data_frame()
# #         print(pred_df)

# #         predict_pipeline = PredictPipeline()
# #         results = predict_pipeline.predict(pred_df)
        
# #         return render_template('home.html', results=f'Prediction based on your inputs is ${results[0]:.2f}')

# # if __name__ == "__main__":
# #     app.run(debug=True, host="0.0.0.0", port=8080)
   
    

# from flask import Flask, request, render_template
# import numpy as np
# import pandas as pd
# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# application = Flask(__name__)
# app = application

# # Route for a home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predictdata', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('home.html')
#     else:
#         # Debug: Print the form data
#         print("Form data received:", request.form)

#         # Validate inputs and provide defaults if None
#         try:
#             data = CustomData(
#                 Item_Weight=request.form.get("Item_Weight") or "0",  # Default to 0 if None
#                 Item_Fat_Content=request.form.get('Item_Fat_Content') or "Low Fat",
#                 Item_Type=request.form.get('Item_Type') or "Others",
#                 Item_MRP=request.form.get('Item_MRP') or "0",
#                 Outlet_Type=request.form.get('Outlet_Type') or "Supermarket",
#                 Outlet_Age=request.form.get('Outlet_Age') or "0"
#             )
#         except ValueError as e:
#             return render_template('home.html', results=f"Error: Invalid input - {str(e)}")

#         pred_df = data.get_data_as_data_frame()
#         print("Prediction DataFrame:", pred_df)

#         predict_pipeline = PredictPipeline()
#         try:
#             results = predict_pipeline.predict(pred_df)
#             return render_template('home.html', results=f'Predicted Sales: {results[0]:,.2f} UGX')
#         except Exception as e:
#             return render_template('home.html', results=f"Prediction Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=8080)