"""Reproducible Olist training pipeline. Expects the original 9 CSVs in data/olist."""
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             f1_score, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (ROOT, MODEL_DIR, MODEL_PATH, METADATA_PATH, FEATURES,
                     NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def build_olist_dataset(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    required = ["olist_orders_dataset.csv", "olist_customers_dataset.csv", "olist_order_items_dataset.csv"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError("Arquivos Olist obrigatórios ausentes: " + ", ".join(missing))

    orders = pd.read_csv(data_dir / required[0], parse_dates=["order_purchase_timestamp",
        "order_estimated_delivery_date", "order_delivered_customer_date"])
    customers = pd.read_csv(data_dir / required[1], dtype={"customer_zip_code_prefix": str})
    items = pd.read_csv(data_dir / required[2])

    # Aggregate to one row per order before joining to avoid fan-out duplication.
    item_agg = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )
    data = orders.merge(customers[["customer_id", "customer_state", "customer_zip_code_prefix"]],
                        on="customer_id", how="left", validate="many_to_one")
    data = data.merge(item_agg, on="order_id", how="left", validate="one_to_one")
    data = data[(data.order_status == "delivered") & data.order_purchase_timestamp.notna()
                & data.order_estimated_delivery_date.notna() & data.order_delivered_customer_date.notna()].copy()
    # The target is calendar-day lateness, not timestamp lateness.
    data["target"] = (data.order_delivered_customer_date.dt.normalize() >
                      data.order_estimated_delivery_date.dt.normalize()).astype(int)
    data["promised_days"] = (data.order_estimated_delivery_date.dt.normalize() -
                             data.order_purchase_timestamp.dt.normalize()).dt.days
    data = data[data.promised_days.ge(0)].copy()
    data["purchase_month"] = data.order_purchase_timestamp.dt.month
    data["purchase_weekday"] = data.order_purchase_timestamp.dt.weekday
    data["purchase_hour"] = data.order_purchase_timestamp.dt.hour
    data["customer_state"] = data.customer_state.astype("string").str.upper()
    data["zip_prefix"] = data.customer_zip_code_prefix.astype("string").str.zfill(5).str[:5]
    data = data.dropna(subset=["item_count", "total_price", "total_freight"])
    return data.sort_values("order_purchase_timestamp").reset_index(drop=True)


def make_pipeline(model):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", prep), ("classifier", model)])


def _metrics(y, scores, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "prevalence": float(np.mean(y)),
    }


def train(data_dir: Path = ROOT / "data" / "olist"):
    data = build_olist_dataset(data_dir)
    if len(data) < 1000 or data.target.nunique() < 2:
        raise ValueError("Base elegível insuficiente para treinamento confiável.")
    X, y = data[FEATURES].copy(), data.target
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    n = len(data)
    train_end, val_end = int(n * .70), int(n * .85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    fitted, val_results = {}, {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_val)[:, 1]
        fitted[name] = pipe
        val_results[name] = _metrics(y_val, scores)
    winner = max(val_results, key=lambda k: val_results[k]["pr_auc"])

    # Refit selected pipeline on train + validation; keep the temporal test untouched.
    final_model = make_pipeline(candidates[winner])
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_model.fit(X_trainval, y_trainval)
    test_scores = final_model.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test, test_scores)

    # Lift in top 20%: compare actual delay rate in highest-scored quintile with test prevalence.
    test_rank = pd.DataFrame({"score": test_scores, "target": y_test.to_numpy()}).sort_values("score", ascending=False)
    top = test_rank.head(max(1, int(np.ceil(len(test_rank) * .20))))
    test_metrics["top_20pct_delay_rate"] = float(top.target.mean())
    test_metrics["top_20pct_lift"] = float(top.target.mean() / y_test.mean()) if y_test.mean() else None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    metadata = {
        "model_name": winner, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(X_trainval)), "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)), "train_period": [str(data.order_purchase_timestamp.iloc[0]), str(data.order_purchase_timestamp.iloc[train_end-1])],
        "test_period": [str(data.order_purchase_timestamp.iloc[val_end]), str(data.order_purchase_timestamp.iloc[-1])],
        "validation_metrics": val_results, "test_metrics": test_metrics,
        "features": FEATURES, "target": "delivery_after_promised_calendar_day",
        "warning": "Academic Olist model; not validated for current Mercado Livre operations."
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def load_trusted_model():
    """Load only the locally generated artifact; never accept a model upload."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return joblib.load(MODEL_PATH), metadata
