import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import model_regression
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.read_csv("input/processed_data.csv")
    test = pd.read_csv("input/processed_test.csv")
    ids = test["cellid"].values
    
    targets = ["order", "order_within_phase"]
    categorical_cols = ["phase"]

    submission = pd.DataFrame()
    submission["cellid"] = ids

    for target in targets:
        print(f"\nTraining model for {target}:")
        
        X = df.drop(columns=["cellid"] + targets)
        y = df[target]
        X_test = test.drop(columns=["cellid"] + targets)

        encoder = TargetEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols], y)
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        model = model_regression.get_model()
        model.fit(X, y)
        
        train_preds = model.predict(X)
        test_preds = model.predict(X_test)
        
        print(f"RMSE on train: {np.sqrt(mean_squared_error(y, train_preds)):.4f}")
        submission[target] = test_preds

    submission.to_csv("output/submission_reg.csv", index=False)