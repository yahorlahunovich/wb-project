import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import model_regression
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.read_csv("input/processed_data.csv")
    df.drop(columns=["cellid", "phase"], inplace=True)

    targets = ["order", "order_within_phase"]
    
    for target in targets:
        print(f"\nTraining model for {target}:")
        
        X = df.drop(columns=targets)
        y = df[target]

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []

        for fold, (train_index, test_index) in enumerate(kfold.split(X)):
            X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = model_regression.get_model()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

        print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
        print(f"Mean R2: {np.mean(r2_scores):.4f}")
