import pandas as pd

def run_prediction(uploaded_file, model_type):
    df = pd.read_csv(uploaded_file)
    
    if model_type == "Time Series":
        return df.describe()
    elif model_type == "Deep Learning":
        return df.head()
    else:
        return "Unsupported model type."
