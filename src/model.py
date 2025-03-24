from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import catboost as cb

def get_model():
    xgb_model = XGBClassifier(n_estimators=400, learning_rate=0.01, max_depth=5, n_jobs=-1)
    lr_model = LogisticRegression(penalty="l1", solver="liblinear")
    knn_model = KNeighborsClassifier(n_neighbors=12)
    cat_model = cb.CatBoostClassifier(n_estimators=300, learning_rate=0.01, depth=6, loss_function='MultiClass', verbose=0)

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=-1)
    model = VotingClassifier(estimators=[('xgb', xgb_model), ('lr', lr_model), ('cat', cat_model), ('knn', knn_model), ('rf', rf)], voting='soft')
    return model