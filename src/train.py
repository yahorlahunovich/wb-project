import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import model
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    df = pd.read_csv("input/processed_data.csv")
    df.drop(columns=["cellid", "order_within_phase", "order"], inplace=True)
    print(df.columns)

    phase_dict = {
        "G1": 0,
        "S": 1,
        "G2M": 2,
    }
    df["phase"] = df["phase"].map(phase_dict)

    X = df.drop(columns=["phase"])
    y = df["phase"]

    kfold = StratifiedKFold(n_splits=5)
    accuracies = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = model.get_model()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        accuracies.append(accuracy)
        print(f"Fold {fold + 1} accuracy: {accuracy}")

    print(f"Mean accuracy: {np.mean(accuracies)}")