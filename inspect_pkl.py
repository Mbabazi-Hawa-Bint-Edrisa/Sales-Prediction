import joblib

# Load the model
model = joblib.load(r'C:\Users\USER\Desktop\sales pred\artifacts\model.pkl')
print("Model type:", type(model))

# Check if it's a pipeline and print the steps
if hasattr(model, 'steps'):
    print("Pipeline steps:", model.steps)