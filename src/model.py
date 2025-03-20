from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_model():
    # model = XGBClassifier()
    model = LogisticRegression()
    return model