import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import model_classificatiom
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.read_csv("input/processed_data.csv")
    test = pd.read_csv("input/processed_test.csv")
    ids = test["cellid"].values
    df.drop(columns=["cellid", "order_within_phase", "order"], inplace=True)
    test.drop(columns=["cellid", "order_within_phase", "order", "phase"], inplace=True)
    phase_dict = {
        "G1": 0,
        "S": 1,
        "G2M": 2,
    }
    df["phase"] = df["phase"].map(phase_dict)

    X = df.drop(columns=["phase"])
    # print(X.columns)
    # print(test.columns)
    y = df["phase"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    test = scaler.transform(test)

    clf = model_classificatiom.get_model()
    print("Model is training")
    clf.fit(X, y)
    print("Model trained")
    preds = clf.predict(test)

    preds = np.where(preds == 0, "G1", np.where(preds == 1, "S", "G2M"))

    submission = pd.DataFrame({"cellid": ids, "phase": preds})
    submission.to_csv("output/submission.csv", index=False)