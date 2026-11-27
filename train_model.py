from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "final_ebay_data.csv"
MODEL_PATH = BASE_DIR / "used_laptop_estimator.joblib"

TARGET = "Price ($)"
RANDOM_STATE = 42

NUMERIC_COLS = [
    "Screen Size (in)",
    "Ram Size (GB)",
    "SSD Capacity (GB)",
    "Processor Speed (GHz)",
    "Hard Drive Capacity (GB)",
]

CATEGORICAL_COLS = [
    "Brand",
    "Processor",
    "GPU",
    "Resolution Size",
    "OS",
    "Storage Type",
    "Condition",
]


def main():
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset shape:", df.shape)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=5.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }
    results = []
    for name, regressor in models.items():
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", regressor),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        results.append({
            "Model": name,
            "MAE ($)": round(mean_absolute_error(y_test, pred), 2),
            "RMSE ($)": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
            "R2": round(r2_score(y_test, pred), 3),
        })

    results_df = pd.DataFrame(results).sort_values("MAE ($)")
    print("\n=== Model Comparison on Test Set ===")
    print(results_df.to_string(index=False))
    rf_pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "model__max_depth": [6, 8, 12, None],
        "model__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="neg_mean_absolute_error")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    print("\nBest Tuned Parameters:", search.best_params_)
    print("Train MAE ($):", round(mean_absolute_error(y_train, train_pred), 2),
          "| Train R2:", round(r2_score(y_train, train_pred), 3))
    print("Test  MAE ($):", round(mean_absolute_error(y_test, test_pred), 2),
          "| Test RMSE ($):", round(np.sqrt(mean_squared_error(y_test, test_pred)), 2),
          "| Test R2:", round(r2_score(y_test, test_pred), 3))
    joblib.dump(best_model, MODEL_PATH)
    print("\nSaved final pipeline to:", MODEL_PATH)


if __name__ == "__main__":
    main()