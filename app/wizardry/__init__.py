# Dictionary containing available models and their configurations
model_dict = {
    "classification": {
        "Logistic Regression": "sklearn.linear_model.LogisticRegression",
        "Random Forest": "sklearn.ensemble.RandomForestClassifier",
        "SVM": "sklearn.svm.SVC",
        "XGBoost": "xgboost.XGBClassifier"
    },
    "regression": {
        "Linear Regression": "sklearn.linear_model.LinearRegression",
        "Random Forest": "sklearn.ensemble.RandomForestRegressor",
        "SVM": "sklearn.svm.SVR",
        "XGBoost": "xgboost.XGBRegressor"
    },
    "clustering": {
        "K-Means": "sklearn.cluster.KMeans",
        "DBSCAN": "sklearn.cluster.DBSCAN",
        "Hierarchical": "sklearn.cluster.AgglomerativeClustering"
    }
}

# Import the step modules
from . import step4_task_detection
from . import step5_model_selection
from . import step6_training
from . import step7_inference
from . import step8_explainability
from . import step9_outputs

# Create a class to wrap the run function
class Step:
    def __init__(self, module):
        self.module = module
    
    def run(self):
        self.module.run()

# Create step objects
step4_task_detection = Step(step4_task_detection)
step5_model_selection = Step(step5_model_selection)
step6_training = Step(step6_training)
step7_inference = Step(step7_inference)
step8_explainability = Step(step8_explainability)
step9_outputs = Step(step9_outputs)

__all__ = [
    'model_dict',
    'step4_task_detection',
    'step5_model_selection',
    'step6_training',
    'step7_inference',
    'step8_explainability',
    'step9_outputs'
]
