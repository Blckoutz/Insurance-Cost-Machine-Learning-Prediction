# ===========================
# Insurance Cost ML - Master Script
# ===========================
# Prereqs (install in your virtual env):
#   pip install pandas numpy matplotlib scikit-learn xgboost statsmodels
# Optional:
#   pip install joblib

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from xgboost import XGBRegressor as xgb

# ---------------------------
# Config / Canonical Mappings
# ---------------------------
REQUIRED_COLS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

VALID_SEX = {
    "male": "male", "m": "male",
    "female": "female", "f": "female"
}
VALID_SMOKER = {
    "yes": "yes", "y": "yes", "true": "yes", "1": "yes",
    "no": "no", "n": "no", "false": "no", "0": "no"
}
VALID_REGION = {
    "southwest": "southwest", "sw": "southwest",
    "southeast": "southeast", "se": "southeast",
    "northwest": "northwest", "nw": "northwest",
    "northeast": "northeast", "ne": "northeast"
}

# ---------------------------
# Utilities
# ---------------------------
def _iqr_winsorize(series: pd.Series, low_q=0.01, high_q=0.99) -> pd.Series:
    """Clip extreme tails at given quantiles (simple robust cap)."""
    lo, hi = series.quantile([low_q, high_q]).values
    return series.clip(lo, hi)

def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _strip_lower(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _normalize_map(s: pd.Series, mapping: dict, default=np.nan) -> pd.Series:
    return _strip_lower(s).map(mapping).fillna(default)

def validate_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def basic_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=np.number)
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_unique": df.nunique()
    })
    # add numeric min/max where applicable
    rep["min"] = pd.Series({c: num[c].min() for c in num.columns})
    rep["max"] = pd.Series({c: num[c].max() for c in num.columns})
    return rep

# ---------------------------
# Data Cleaning / Extraction
# ---------------------------
def clean_insurance(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    End-to-end cleaning for the insurance dataset:
      - schema validation
      - trim whitespace, normalize categories
      - coerce numerics, drop NAs in required fields
      - plausible range checks
      - (no duplicate removal — survey responses may legitimately repeat)
      - winsorize heavy tails (age, bmi, charges)
    """
    df = df.copy()
    validate_schema(df)

    # Trim whitespace in object cols
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Normalize category labels
    df["sex"]    = _normalize_map(df["sex"], VALID_SEX)
    df["smoker"] = _normalize_map(df["smoker"], VALID_SMOKER)
    df["region"] = _normalize_map(df["region"], VALID_REGION)

    # Coerce numerics
    df = _coerce_numeric(df, ["age", "bmi", "children", "charges"])

    # Drop rows with missing required fields
    df = df.dropna(subset=REQUIRED_COLS)

    # Remove impossible values
    df = df[(df["age"] >= 0) & (df["bmi"] > 0) & (df["children"] >= 0) & (df["charges"] >= 0)]

    # Plausible ranges (adjust as needed)
    df = df[df["age"].between(0, 100)]
    df = df[df["bmi"].between(10, 80)]

    # Ensure integer children
    df["children"] = df["children"].round().astype(int)

    # NOTE: no df.drop_duplicates() here — keep all survey entries

    # Winsorize heavy tails
    for c in ["age", "bmi", "charges"]:
        df[c] = _iqr_winsorize(df[c], 0.01, 0.99)

    if verbose:
        print("Cleaned dataset shape:", df.shape)
        print(basic_quality_report(df))
        cats = {c: sorted(df[c].dropna().unique().tolist()) for c in ["sex", "smoker", "region"]}
        print("Category levels:", cats)

    return df

# ---------------------------
# Pipeline Pieces
# ---------------------------
def make_splits(df: pd.DataFrame, target="charges", test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def make_preprocessor(categorical, numeric):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough"
    )

def make_model(**kwargs):
    """
    XGBRegressor defaults tuned lightly for small/medium tabular sets.
    You can override with: make_model(objective="reg:absoluteerror", max_depth=5, ...)
    """
    params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        # objective defaults to 'reg:squarederror' if not given
        random_state=42
    )
    params.update(kwargs)
    return xgb(**params)

def make_pipeline(preprocessor, model):
    from sklearn.pipeline import Pipeline
    return Pipeline([("preprocessor", preprocessor),
                     ("model", model)])

# ---------------------------
# Target Transform Helpers
# ---------------------------
def transform_target(y: pd.Series, use_log: bool):
    """Apply log1p if requested; otherwise return y unchanged."""
    return np.log1p(y) if use_log else y

def inverse_target(yhat: np.ndarray, used_log: bool):
    """Invert the transform back to original $ scale for evaluation/prints."""
    return np.expm1(yhat) if used_log else yhat

# ---------------------------
# Metrics
# ---------------------------
def evaluate_regression(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ---------------------------
# Visuals
# ---------------------------
def plot_actual_vs_pred(y_true, y_pred, title="Actual vs. Predicted"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title)
    plt.tight_layout(); plt.show()

def plot_residuals(y_true, y_pred, title="Residuals vs. Fitted"):
    residuals = y_true - y_pred
    plt.figure(figsize=(7,4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, ls="--"); plt.xlabel("Predicted"); plt.ylabel("Residual (y - ŷ)")
    plt.title(title); plt.tight_layout(); plt.show()

def plot_feature_importance(pipeline, X_sample):
    """Bar chart for XGB feature importances with recovered feature names."""
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    ohe = pre.named_transformers_["cat"]
    cat_cols = pre.transformers_[0][2]
    cat_names = ohe.get_feature_names_out(cat_cols)
    num_cols = [c for c in X_sample.columns if c not in cat_cols]
    feature_names = np.r_[cat_names, num_cols]

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("No feature_importances_ available.")
        return

    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(9,5))
    plt.bar(range(len(importances)), importances[order])
    plt.xticks(range(len(importances)), feature_names[order], rotation=45, ha="right")
    plt.ylabel("Importance"); plt.title("XGB Feature Importance")
    plt.tight_layout(); plt.show()

# ---------------------------
# Cost Function (Objective) Reporting
# ---------------------------
def _objective_human_readable(objective: str, used_log: bool) -> str:
    """Readable description of objective + whether target was logged."""
    obj_map = {
        "reg:squarederror": "Mean Squared Error (MSE)",
        "reg:absoluteerror": "Mean Absolute Error (L1/MAE)",
        "reg:pseudohubererror": "Pseudo-Huber Loss (robust)",
        "reg:squaredlogerror": "Squared Log Error (MSE on log(y))",
        "count:poisson": "Poisson (count regression)",
        "reg:gamma": "Gamma (right-skewed positive targets)"
    }
    base = obj_map.get(objective, f"{objective} (custom/other)")
    if used_log:
        return f"{base} — trained on log1p(charges), predictions back-transformed to $"
    else:
        return f"{base} — trained on original $ scale"

def get_pipeline_objective(pipeline, used_log_target: bool) -> str:
    """Fetch the XGBoost objective from the fitted Pipeline and describe it."""
    model = pipeline.named_steps["model"]
    objective = model.get_xgb_params().get("objective", "reg:squarederror")
    return _objective_human_readable(objective, used_log_target)

# ---------------------------
# Optional: Statsmodels OLS Baseline
# ---------------------------
def baseline_ols(df: pd.DataFrame, target="charges", use_log=False):
    """OLS baseline with optional log1p(target) for interpretability checks."""
    y = df[target].values
    if use_log:
        y = np.log1p(y)
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    X = sm.add_constant(X)
    ols = sm.OLS(y, X).fit()
    print(ols.summary())
    return ols

# ---------------------------
# User Input → Prediction
# ---------------------------
def _norm_text(s: str) -> str:
    return str(s).strip().lower()

def parse_user_row(age, sex, bmi, children, smoker, region) -> dict:
    """
    Convert raw inputs to a single row dict matching training schema.
    Raises ValueError if inputs are invalid.
    """
    # Numeric validation
    try:
        age = int(float(age))
        bmi = float(bmi)
        children = int(float(children))
    except Exception:
        raise ValueError("age, bmi, and children must be numeric (children/age as whole numbers).")

    if not (0 <= age <= 100):
        raise ValueError("age must be between 0 and 100.")
    if not (10 <= bmi <= 80):
        raise ValueError("bmi must be between 10 and 80.")
    if children < 0:
        raise ValueError("children must be ≥ 0.")

    # Categorical normalization
    sex_in = VALID_SEX.get(_norm_text(sex))
    if sex_in is None:
        raise ValueError("sex must be one of: male/female (or m/f).")

    smoker_in = VALID_SMOKER.get(_norm_text(smoker))
    if smoker_in is None:
        raise ValueError("smoker must be yes/no (y/n, 1/0, true/false).")

    region_in = VALID_REGION.get(_norm_text(region))
    if region_in is None:
        raise ValueError("region must be one of: northeast/northwest/southeast/southwest (or ne/nw/se/sw).")

    return {
        "age": age,
        "sex": sex_in,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_in,
        "region": region_in
    }

def predict_from_inputs(pipeline, use_log_target: bool, **kwargs) -> tuple[float, str]:
    """
    kwargs: age, sex, bmi, children, smoker, region
    Returns: (predicted_charge_dollars, cost_function_text)
    """
    row = parse_user_row(**kwargs)
    X_new = pd.DataFrame([row])
    yhat_t = pipeline.predict(X_new)                   # prediction on transformed target if used
    yhat = inverse_target(yhat_t, use_log_target)      # back to $ scale
    cost_fn = get_pipeline_objective(pipeline, use_log_target)
    return float(yhat[0]), cost_fn

def interactive_prompt(pipeline, use_log_target: bool):
    """
    Simple CLI loop to collect user inputs and print prediction + cost function.
    Press ENTER on 'age' to exit.
    """
    print("\n--- Insurance Cost Predictor ---")
    print("Press ENTER at 'age' to quit.")
    print("Cost function:", get_pipeline_objective(pipeline, use_log_target))
    while True:
        age = input("\nAge (0–100): ").strip()
        if age == "":
            print("Goodbye!")
            break
        sex      = input("Sex (male/female): ").strip()
        bmi      = input("BMI (10–80): ").strip()
        children = input("Children (0+): ").strip()
        smoker   = input("Smoker? (yes/no): ").strip()
        region   = input("Region (northeast/northwest/southeast/southwest or ne/nw/se/sw): ").strip()
        try:
            pred, cost_fn = predict_from_inputs(
                pipeline, use_log_target,
                age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region
            )
            print(f"Estimated annual charge: ${pred:,.2f}")
            print(f"Cost function used: {cost_fn}")
        except ValueError as e:
            print(f"[Input error] {e}")
        except Exception as e:
            print(f"[Unexpected error] {e}")

# ---------------------------
# Train + Evaluate + Serve
# ---------------------------
def train_predictor(csv_path="insurance.csv", USE_LOG_TARGET=True, verbose_clean=True, **model_overrides):
    # Load & clean
    raw = pd.read_csv(csv_path)
    df = clean_insurance(raw, verbose=verbose_clean)

    # Split
    X_train, X_test, y_train, y_test = make_splits(df, target="charges", test_size=0.2, seed=42)

    # Pipeline
    categorical = ["sex", "smoker", "region"]
    numeric     = ["age", "bmi", "children"]
    pre   = make_preprocessor(categorical, numeric)
    model = make_model(**model_overrides)
    pipe  = make_pipeline(pre, model)

    # Fit on (optionally) log target
    y_train_t = transform_target(y_train, USE_LOG_TARGET)
    pipe.fit(X_train, y_train_t)

    # Predict, inverse-transform to $
    y_pred_t = pipe.predict(X_test)
    y_pred   = inverse_target(y_pred_t, USE_LOG_TARGET)

    # Metrics
    metrics = evaluate_regression(y_test, y_pred)
    print("\n=== Test Metrics (original $ scale) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:,.4f}")
    print("Cost function:", get_pipeline_objective(pipe, USE_LOG_TARGET))

    # Plots
    plot_actual_vs_pred(y_test, y_pred, title="Actual vs. Predicted (Test)")
    plot_residuals(y_test, y_pred, title="Residuals vs. Fitted (Test)")
    plot_feature_importance(pipe, X_train)

    return pipe, USE_LOG_TARGET

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Toggle log-target here (True often helps with skewed costs)
    use_log = True

    # Optional: override objective, depth, etc.
    # e.g., objective="reg:absoluteerror"
    pipeline, used_log = train_predictor(
        csv_path="insurance.csv",
        USE_LOG_TARGET=use_log,
        verbose_clean=True,
        # objective="reg:squarederror",
        # max_depth=5,
        # n_estimators=800,
    )

    # Optional: OLS baseline on same cleaned data (commented out by default)
    # raw = pd.read_csv("insurance.csv")
    # dfc = clean_insurance(raw, verbose=False)
    # baseline_ols(dfc, target="charges", use_log=use_log)

    # Interactive CLI predictions
    interactive_prompt(pipeline, used_log)
