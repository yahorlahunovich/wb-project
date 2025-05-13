from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import catboost as cb
from sklearn.svm import SVC
import lightgbm as lgb

def get_model():
    xgb_model = XGBClassifier(n_estimators=800, learning_rate=0.01, max_depth=5, n_jobs=-1)
    lr_model = LogisticRegression(penalty="l1", solver="liblinear")
    knn_model = KNeighborsClassifier(n_neighbors=12)
    cat_model = cb.CatBoostClassifier(n_estimators=800, learning_rate=0.01, depth=4, loss_function='MultiClass', verbose=0)
    rf = RandomForestClassifier(n_estimators=800, max_depth=4, n_jobs=-1)
    svc_model = model = SVC(gamma='auto', C=0.6, probability=True)
    lgb_model = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.01, max_depth=2, n_jobs=-1, verbosity=-1)
    model = VotingClassifier(estimators=[('xgb', xgb_model), ('lr', lr_model), 
                                         ('cat', cat_model), ('knn', knn_model), 
                                         ('rf', rf), ('svc', svc_model), 
                                         ('lgb', lgb_model)], voting='soft')
    

    return model