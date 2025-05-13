from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import catboost as cb
from sklearn.svm import SVR
import lightgbm as lgb

def get_model():
    xgb_model = XGBRegressor(n_estimators=800, learning_rate=0.01, max_depth=5, n_jobs=1)
    lr_model = Lasso()
    knn_model = KNeighborsRegressor(n_neighbors=12, n_jobs=1)
    cat_model = cb.CatBoostRegressor(n_estimators=800, learning_rate=0.01, depth=4, verbose=0, thread_count=1)
    rf = RandomForestRegressor(n_estimators=800, max_depth=4, n_jobs=1)
    svc_model = SVR(gamma='auto', C=0.6)
    lgb_model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.01, max_depth=2, n_jobs=1, verbosity=-1)
    
 
    model = VotingRegressor(
        estimators=[
            ('xgb', xgb_model), 
            ('lr', lr_model), 
            ('cat', cat_model), 
            ('knn', knn_model), 
            ('rf', rf), 
            ('svc', svc_model), 
            ('lgb', lgb_model)
        ],
        n_jobs=1  
    )
    
    return model