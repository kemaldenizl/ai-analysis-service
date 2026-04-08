import os
import json
import joblib
import pandas as pd
import numpy as np

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "../data/transactions.csv")
MODEL_STORE_PATH = os.path.join(BASE_DIR, "ml/models_store/monthly_forecaster.joblib")
METRICS_STORE_PATH = os.path.join(BASE_DIR, "ml/models_store/monthly_forecaster_metrics.json")

RANDOM_STATE = 42


@dataclass(frozen=True)
class ForecastConfig:
    snapshot_day: int = 15
    min_monthly_transaction_count: int = 5
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    n_splits: int = 3
    random_state: int = RANDOM_STATE


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print("1.) Veri yükleniyor.")
    df = pd.read_csv(file_path)
    print(f"Yükleme tamamlandı. Kayıt sayısı: {len(df)}")

    print("2.) Veri temizleniyor.")
    required_columns = [
        "Date", "Description", "Amount",
        "Transaction Type", "Category", "Account Name"
    ]

    df.columns = df.columns.str.strip()
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik kolon(lar) bulundu: {missing_columns}")

    df = df[required_columns].copy()

    text_columns = ["Description", "Transaction Type", "Category", "Account Name"]
    for col in text_columns:
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df = df.dropna(subset=["Date", "Amount", "Transaction Type", "Category"]).copy()

    df["Transaction Type"] = df["Transaction Type"].str.lower().str.strip()
    df["Category"] = df["Category"].str.lower().str.strip()
    df["Description"] = df["Description"].str.lower().str.strip()
    df["Account Name"] = df["Account Name"].str.lower().str.strip()

    valid_transaction_types = {"debit", "credit"}
    df = df[df["Transaction Type"].isin(valid_transaction_types)].copy()

    df["Amount"] = df["Amount"].abs()
    df = df[df["Amount"] > 0].copy()

    non_income_credit_categories = {
        "credit card payment",
        "transfer",
        "balance adjustment",
        "internal transfer",
        "payment",
    }

    non_income_credit_descriptions = {
        "credit card payment",
        "transfer",
        "payment",
    }

    is_true_income = (
        (df["Transaction Type"] == "credit")
        & (~df["Category"].isin(non_income_credit_categories))
        & (~df["Description"].isin(non_income_credit_descriptions))
    )

    df["Expense"] = np.where(df["Transaction Type"] == "debit", df["Amount"], 0.0)
    df["Income"] = np.where(is_true_income, df["Amount"], 0.0)
    df["Signed Amount"] = np.where(df["Transaction Type"] == "debit", -df["Amount"], df["Amount"])

    df = df.drop_duplicates(
        subset=["Date", "Description", "Amount", "Transaction Type", "Category", "Account Name"]
    ).copy()

    df["Year"] = df["Date"].dt.year.astype(int)
    df["Month"] = df["Date"].dt.month.astype(int)
    df["Day"] = df["Date"].dt.day.astype(int)
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["MonthStartDate"] = df["Date"].dt.to_period("M").dt.start_time
    df["DayOfWeek"] = df["Date"].dt.dayofweek.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Temizleme tamamlandı. Kayıt sayısı: {len(df)}")
    return df


def _safe_div(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return 0.0
    return float(a / b)


def _dominant_category_ratio(expense_df: pd.DataFrame) -> float:
    if expense_df.empty:
        return 0.0
    category_sum = expense_df.groupby("Category")["Amount"].sum()
    total = float(category_sum.sum())
    if total <= 0:
        return 0.0
    return float(category_sum.max() / total)


def _category_entropy(expense_df: pd.DataFrame) -> float:
    if expense_df.empty:
        return 0.0
    category_sum = expense_df.groupby("Category")["Amount"].sum()
    total = float(category_sum.sum())
    if total <= 0:
        return 0.0
    ratios = (category_sum / total).values
    return float(-(ratios * np.log(ratios + 1e-12)).sum())


def _build_monthly_base_table(df: pd.DataFrame) -> pd.DataFrame:
    print("3.) Aylık base table hazırlanıyor.")

    monthly = (
        df.groupby("YearMonth", sort=True)
        .agg(
            MonthStartDate=("MonthStartDate", "min"),
            Year=("Year", "first"),
            Month=("Month", "first"),
            month_total_expense=("Expense", "sum"),
            month_total_income=("Income", "sum"),
            total_transactions=("Amount", "count"),
            expense_transaction_count=("Expense", lambda x: int((x > 0).sum())),
            income_transaction_count=("Income", lambda x: int((x > 0).sum())),
            active_days=("Date", lambda x: x.dt.date.nunique()),
        )
        .reset_index()
        .sort_values("MonthStartDate")
        .reset_index(drop=True)
    )

    monthly["prev_month_total_expense"] = monthly["month_total_expense"].shift(1)
    monthly["prev_2_month_total_expense"] = monthly["month_total_expense"].shift(2)
    monthly["rolling_3m_expense_mean"] = monthly["month_total_expense"].shift(1).rolling(3, min_periods=1).mean()
    monthly["rolling_3m_expense_std"] = monthly["month_total_expense"].shift(1).rolling(3, min_periods=1).std()
    monthly["rolling_3m_income_mean"] = monthly["month_total_income"].shift(1).rolling(3, min_periods=1).mean()

    monthly["expense_momentum"] = monthly["prev_month_total_expense"] - monthly["prev_2_month_total_expense"]
    monthly["expense_vs_prev_month_ratio"] = np.where(
        monthly["prev_month_total_expense"] > 0,
        monthly["month_total_expense"] / monthly["prev_month_total_expense"],
        0.0
    )

    numeric_cols = monthly.select_dtypes(include=[np.number]).columns
    monthly[numeric_cols] = monthly[numeric_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    print(f"Aylık base table hazır. Satır sayısı: {len(monthly)}")
    return monthly


def _build_month_snapshot_row(
    month_df: pd.DataFrame,
    monthly_base_lookup: pd.DataFrame,
    snapshot_day: int
) -> Dict[str, Any]:
    snapshot_df = month_df[month_df["Day"] <= snapshot_day].copy()
    if snapshot_df.empty:
        return {}

    expense_df = snapshot_df[snapshot_df["Transaction Type"] == "debit"].copy()
    income_df = snapshot_df[snapshot_df["Income"] > 0].copy()
    full_expense_df = month_df[month_df["Transaction Type"] == "debit"].copy()

    if full_expense_df.empty:
        return {}

    year_month = month_df["YearMonth"].iloc[0]
    monthly_ref = monthly_base_lookup.loc[monthly_base_lookup["YearMonth"] == year_month]
    if monthly_ref.empty:
        return {}

    monthly_ref = monthly_ref.iloc[0]

    total_expense_so_far = float(expense_df["Amount"].sum())
    total_income_so_far = float(income_df["Amount"].sum())
    expense_tx_count_so_far = int(len(expense_df))
    income_tx_count_so_far = int(len(income_df))
    total_tx_so_far = int(len(snapshot_df))

    active_days_so_far = int(snapshot_df["Date"].dt.date.nunique())
    expense_active_days_so_far = int(expense_df["Date"].dt.date.nunique()) if not expense_df.empty else 0

    avg_expense_so_far = float(expense_df["Amount"].mean()) if not expense_df.empty else 0.0
    std_expense_so_far = float(expense_df["Amount"].std(ddof=0)) if len(expense_df) > 1 else 0.0
    max_expense_so_far = float(expense_df["Amount"].max()) if not expense_df.empty else 0.0

    weekend_expense_so_far = float(expense_df.loc[expense_df["IsWeekend"] == 1, "Amount"].sum()) if not expense_df.empty else 0.0
    unique_categories_so_far = int(expense_df["Category"].nunique()) if not expense_df.empty else 0

    midpoint = max(1, snapshot_day // 2)
    first_half_expense_so_far = float(expense_df.loc[expense_df["Day"] <= midpoint, "Amount"].sum()) if not expense_df.empty else 0.0
    second_half_expense_so_far = float(expense_df.loc[expense_df["Day"] > midpoint, "Amount"].sum()) if not expense_df.empty else 0.0

    row = {
        "YearMonth": year_month,
        "MonthStartDate": month_df["MonthStartDate"].iloc[0],
        "Year": int(month_df["Year"].iloc[0]),
        "Month": int(month_df["Month"].iloc[0]),
        "snapshot_day": int(snapshot_day),

        "total_expense_so_far": total_expense_so_far,
        "expense_tx_count_so_far": expense_tx_count_so_far,
        "avg_daily_expense_so_far": _safe_div(total_expense_so_far, snapshot_day),
        "avg_expense_so_far": avg_expense_so_far,
        "std_expense_so_far": std_expense_so_far,
        "max_expense_so_far": max_expense_so_far,

        "total_income_so_far": total_income_so_far,
        "income_tx_count_so_far": income_tx_count_so_far,
        "expense_to_income_ratio_so_far": _safe_div(total_expense_so_far, total_income_so_far),

        "total_tx_so_far": total_tx_so_far,
        "expense_frequency_ratio_so_far": _safe_div(expense_tx_count_so_far, total_tx_so_far),

        "active_days_ratio_so_far": _safe_div(active_days_so_far, snapshot_day),
        "expense_active_days_ratio_so_far": _safe_div(expense_active_days_so_far, snapshot_day),

        "weekend_expense_ratio_so_far": _safe_div(weekend_expense_so_far, total_expense_so_far),
        "unique_categories_so_far": unique_categories_so_far,
        "dominant_category_ratio_so_far": _dominant_category_ratio(expense_df),
        "category_entropy_so_far": _category_entropy(expense_df),

        "expense_trend_delta_so_far": second_half_expense_so_far - first_half_expense_so_far,
        "expense_trend_ratio_so_far": _safe_div(second_half_expense_so_far, first_half_expense_so_far),

        "prev_month_total_expense": float(monthly_ref["prev_month_total_expense"]),
        "prev_2_month_total_expense": float(monthly_ref["prev_2_month_total_expense"]),
        "rolling_3m_expense_mean": float(monthly_ref["rolling_3m_expense_mean"]),
        "rolling_3m_expense_std": float(monthly_ref["rolling_3m_expense_std"]),
        "rolling_3m_income_mean": float(monthly_ref["rolling_3m_income_mean"]),
        "expense_momentum": float(monthly_ref["expense_momentum"]),

        "month_total_expense_target": float(full_expense_df["Amount"].sum()),
    }

    return row


def build_forecast_dataset(
    df: pd.DataFrame,
    snapshot_day: int = 15,
    min_monthly_transaction_count: int = 5,
) -> pd.DataFrame:
    print("4.) Forecast dataset hazırlanıyor.")

    monthly_base = _build_monthly_base_table(df)

    rows: List[Dict[str, Any]] = []

    grouped = df.groupby("YearMonth", sort=True)
    for _, month_df in grouped:
        month_df = month_df.sort_values("Date").copy()

        if len(month_df) < min_monthly_transaction_count:
            continue

        debit_count = int((month_df["Transaction Type"] == "debit").sum())
        if debit_count == 0:
            continue

        row = _build_month_snapshot_row(
            month_df=month_df,
            monthly_base_lookup=monthly_base,
            snapshot_day=snapshot_day,
        )
        if row:
            rows.append(row)

    forecast_df = pd.DataFrame(rows)
    if forecast_df.empty:
        raise ValueError("Forecast dataset oluşturulamadı.")

    forecast_df = forecast_df.sort_values("MonthStartDate").reset_index(drop=True)

    numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
    forecast_df[numeric_cols] = (
        forecast_df[numeric_cols]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    print(f"Forecast dataset hazırlandı. Aylık örnek sayısı: {len(forecast_df)}")
    print(f"Toplam kolon sayısı: {forecast_df.shape[1]}")
    return forecast_df


def split_time_based_data(
    forecast_df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("5.) Time-based split yapılıyor.")

    ordered_df = forecast_df.sort_values("MonthStartDate").reset_index(drop=True)
    n = len(ordered_df)

    if n < 10:
        raise ValueError(f"Forecast modeli için yeterli aylık örnek yok. Örnek sayısı: {n}")

    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = ordered_df.iloc[:train_end].copy()
    valid_df = ordered_df.iloc[train_end:valid_end].copy()
    test_df = ordered_df.iloc[valid_end:].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Train / valid / test split başarısız.")

    print(f"Train boyutu: {len(train_df)}")
    print(f"Valid boyutu: {len(valid_df)}")
    print(f"Test boyutu : {len(test_df)}")

    return train_df, valid_df, test_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )


def build_candidate_models(config: ForecastConfig) -> Dict[str, Any]:
    return {
        "baseline_linear": LinearRegression(),
        "ridge": Ridge(alpha=3.0),
        "elasticnet": ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=config.random_state, max_iter=10000),
        "huber": HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500),
        "random_forest_small": RandomForestRegressor(
            n_estimators=150,
            max_depth=4,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=config.random_state,
            n_jobs=-1,
        ),
    }


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_baseline(train_y: pd.Series, eval_y: pd.Series) -> Dict[str, float]:
    baseline_pred = np.full(len(eval_y), fill_value=float(train_y.mean()))
    return calculate_regression_metrics(eval_y.to_numpy(), baseline_pred)


def run_time_series_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Dict[str, Any],
    n_splits: int,
) -> Dict[str, Dict[str, float]]:
    print("6.) TimeSeries CV başlıyor.")
    results: Dict[str, Dict[str, float]] = {}

    if len(X_train) < (n_splits + 2):
        n_splits = max(2, min(3, len(X_train) - 1))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for model_name, model in models.items():
        rmse_scores = []
        mae_scores = []
        r2_scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor(X_tr)),
                ("model", clone(model)),
            ])

            pipeline.fit(X_tr, y_tr)
            pred = pipeline.predict(X_va)
            metrics = calculate_regression_metrics(y_va.to_numpy(), pred)

            rmse_scores.append(metrics["rmse"])
            mae_scores.append(metrics["mae"])
            r2_scores.append(metrics["r2"])

            print(
                f"{model_name} | fold={fold_idx} | "
                f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}"
            )

        results[model_name] = {
            "cv_rmse_mean": float(np.mean(rmse_scores)),
            "cv_mae_mean": float(np.mean(mae_scores)),
            "cv_r2_mean": float(np.mean(r2_scores)),
        }

    return results


def train_monthly_forecast_model(
    forecast_df: pd.DataFrame,
    config: ForecastConfig,
) -> Dict[str, Any]:
    print("7.) Forecast regression model eğitimi başlıyor.")

    target_col = "month_total_expense_target"
    drop_cols = ["YearMonth", "MonthStartDate", target_col]

    train_df, valid_df, test_df = split_time_based_data(
        forecast_df=forecast_df,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
    )

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_col]

    X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
    y_valid = valid_df[target_col]

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[target_col]

    models = build_candidate_models(config)

    cv_results = run_time_series_cv(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        models=models,
        n_splits=config.n_splits,
    )

    metrics_summary: Dict[str, Any] = {}

    baseline_valid = evaluate_baseline(y_train, y_valid)
    baseline_test = evaluate_baseline(pd.concat([y_train, y_valid], axis=0), y_test)

    metrics_summary["baseline_mean_valid"] = baseline_valid
    metrics_summary["baseline_mean_test"] = baseline_test

    best_model_name = None
    best_valid_rmse = float("inf")

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", clone(model)),
        ])

        pipeline.fit(X_train, y_train)
        valid_pred = pipeline.predict(X_valid)

        valid_metrics = calculate_regression_metrics(y_valid.to_numpy(), valid_pred)
        metrics_summary[f"{model_name}_valid"] = valid_metrics

        print(
            f"{model_name} | VALID | "
            f"RMSE={valid_metrics['rmse']:.4f} | "
            f"MAE={valid_metrics['mae']:.4f} | "
            f"R2={valid_metrics['r2']:.4f}"
        )

        if valid_metrics["rmse"] < best_valid_rmse:
            best_valid_rmse = valid_metrics["rmse"]
            best_model_name = model_name

    if best_model_name is None:
        raise ValueError("En iyi forecast modeli seçilemedi.")

    print(f"Seçilen en iyi model: {best_model_name}")

    X_train_valid = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
    y_train_valid = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

    final_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train_valid)),
        ("model", clone(models[best_model_name])),
    ])

    final_pipeline.fit(X_train_valid, y_train_valid)

    test_pred = final_pipeline.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test.to_numpy(), test_pred)
    metrics_summary[f"{best_model_name}_test"] = test_metrics

    print(
        f"{best_model_name} | TEST | "
        f"RMSE={test_metrics['rmse']:.4f} | "
        f"MAE={test_metrics['mae']:.4f} | "
        f"R2={test_metrics['r2']:.4f}"
    )

    model_artifact = {
        "model_type": "month_end_expense_forecaster",
        "model_version": "2.0.0",
        "best_model_name": best_model_name,
        "model": final_pipeline,
        "feature_columns": list(X_train_valid.columns),
        "target_column": target_col,
        "snapshot_day": config.snapshot_day,
        "train_shape": X_train_valid.shape,
        "metrics": metrics_summary,
        "cv_results": cv_results,
        "config": asdict(config),
    }

    return model_artifact


def print_forecast_report(model_artifact: Dict[str, Any]) -> None:
    print("\n=== FORECAST / REGRESSION REPORT ===")
    print(f"Model type       : {model_artifact['model_type']}")
    print(f"Model version    : {model_artifact['model_version']}")
    print(f"Best model       : {model_artifact['best_model_name']}")
    print(f"Snapshot day     : {model_artifact['snapshot_day']}")
    print(f"Train shape      : {model_artifact['train_shape']}")

    print("\n--- CV Results ---")
    for model_name, result in model_artifact["cv_results"].items():
        print(
            f"{model_name:<20} | "
            f"CV RMSE={result['cv_rmse_mean']:.4f} | "
            f"CV MAE={result['cv_mae_mean']:.4f} | "
            f"CV R2={result['cv_r2_mean']:.4f}"
        )

    print("\n--- Metrics Summary ---")
    for key, value in model_artifact["metrics"].items():
        print(
            f"{key:<24} | "
            f"RMSE={value['rmse']:.4f} | "
            f"MAE={value['mae']:.4f} | "
            f"R2={value['r2']:.4f}"
        )


if __name__ == "__main__":
    try:
        config = ForecastConfig(
            snapshot_day=15,
            min_monthly_transaction_count=5,
            train_ratio=0.70,
            valid_ratio=0.15,
            n_splits=3,
        )

        raw_data = load_and_clean_data(DATA_PATH)

        forecast_dataset = build_forecast_dataset(
            df=raw_data,
            snapshot_day=config.snapshot_day,
            min_monthly_transaction_count=config.min_monthly_transaction_count,
        )

        model_artifact = train_monthly_forecast_model(
            forecast_df=forecast_dataset,
            config=config,
        )

        os.makedirs(os.path.dirname(MODEL_STORE_PATH), exist_ok=True)
        joblib.dump(model_artifact, MODEL_STORE_PATH)

        metrics_payload = {
            "model_type": model_artifact["model_type"],
            "model_version": model_artifact["model_version"],
            "best_model_name": model_artifact["best_model_name"],
            "snapshot_day": model_artifact["snapshot_day"],
            "train_shape": model_artifact["train_shape"],
            "metrics": model_artifact["metrics"],
            "cv_results": model_artifact["cv_results"],
            "config": model_artifact["config"],
        }

        with open(METRICS_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        print(f"\nModel başarıyla kaydedildi: {MODEL_STORE_PATH}")
        print(f"Metrikler başarıyla kaydedildi: {METRICS_STORE_PATH}")

        print_forecast_report(model_artifact)

    except Exception as e:
        print(f"HATA: Forecast model eğitimi sırasında bir sorun oluştu: {str(e)}")