from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path("processed")
schema = json.loads((DATA_DIR / "feature_columns.json").read_text())

df = pd.read_csv(
    DATA_DIR / "sp500_model_ready.csv",
    parse_dates=["date", "feature_date"]
)

def get_split(name):
    rows = df.loc[df["split"].eq(name)]
    X = rows[schema["feature_columns"]].copy()
    y = rows[schema["target_column"]].astype("int8").copy()
    return X, y

X_train, y_train = get_split("train")
X_valid, y_valid = get_split("validation")
X_test, y_test = get_split("test")

print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.value_counts(normalize=True))

#fit only within train set
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

#evaluation function
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def evaluate_model(y_true, y_pred, y_score, model_name):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

#mlp model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,                 # L2 正则
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,        # 从训练集内部切 10% 做早停，不是外部 validation
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
)

mlp.fit(X_train_scaled, y_train)

y_valid_pred = mlp.predict(X_valid_scaled)
y_valid_score = mlp.predict_proba(X_valid_scaled)[:, 1]

valid_result = evaluate_model(y_valid, y_valid_pred, y_valid_score, "MLP_baseline")
print(valid_result)
print(classification_report(y_valid, y_valid_pred, digits=4))

#parameter tuning
from itertools import product

param_grid = [
    {"hidden_layer_sizes": (32,),        "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 32),     "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 32),     "alpha": 1e-3},
    {"hidden_layer_sizes": (128, 64),    "alpha": 1e-4},
    {"hidden_layer_sizes": (128, 64),    "alpha": 1e-3},
]

results = []
best = None

for params in param_grid:
    model = MLPClassifier(
        **params,
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_valid_scaled)
    y_score = model.predict_proba(X_valid_scaled)[:, 1]
    res = evaluate_model(y_valid, y_pred, y_score, f"MLP_{params}")
    results.append(res)

    if best is None or res["auc"] > best["result"]["auc"]:
        best = {"params": params, "model": model, "result": res}

print("Best params:", best["params"])
print("Best validation:", best["result"])

#train + validation
X_train_final = pd.concat([X_train, X_valid], axis=0)
y_train_final = pd.concat([y_train, y_valid], axis=0)

scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_test_scaled = scaler_final.transform(X_test)

final_mlp = MLPClassifier(
    **best["params"],
    activation="relu",
    solver="adam",
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
)

final_mlp.fit(X_train_final_scaled, y_train_final)

y_test_pred = final_mlp.predict(X_test_scaled)
y_test_score = final_mlp.predict_proba(X_test_scaled)[:, 1]

test_result = evaluate_model(y_test, y_test_pred, y_test_score, "MLP_final")
print(test_result)
print(classification_report(y_test, y_test_pred, digits=4))

#compared with dummy benchmark
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

y_dummy_pred = dummy.predict(X_test)
y_dummy_score = dummy.predict_proba(X_test)[:, 1]

dummy_result = evaluate_model(y_test, y_dummy_pred, y_dummy_score, "Majority_baseline")
print(dummy_result)

#saving result
import json
from pathlib import Path

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

all_results = [valid_result, test_result, dummy_result]
(RESULT_DIR / "mlp_results.json").write_text(
    json.dumps(all_results, indent=2)
)

#convert to csv
pd.DataFrame([
    {k: v for k, v in r.items() if k != "confusion_matrix"}
    for r in all_results
]).to_csv(RESULT_DIR / "mlp_results.csv", index=False)