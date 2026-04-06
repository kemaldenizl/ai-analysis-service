import os
import joblib
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "../data/transactions.csv")
MODEL_STORE_PATH = os.path.join(BASE_DIR, "ml/models_store/spending_profiler.joblib")

RANDOM_STATE = 42


@dataclass(frozen=True)
class ProfilingConfig:
    variance_threshold: float = 0.01
    candidate_ks: Tuple[int, ...] = (3, 4)
    min_cluster_ratio: float = 0.05
    outlier_exclusion_enabled: bool = True
    outlier_feature_threshold: int = 3
    outlier_zscore_threshold: float = 3.5
    n_init: int = 40
    max_iter: int = 800
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

    df["Signed Amount"] = np.where(
        df["Transaction Type"] == "debit",
        -df["Amount"],
        df["Amount"]
    )

    df = df.drop_duplicates(
        subset=["Date", "Description", "Amount", "Transaction Type", "Category", "Account Name"]
    ).copy()

    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Temizleme tamamlandı. Kayıt sayısı: {len(df)}")
    return df


def extract_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    print("3.) Weekly Feature Engineering başlıyor.")

    if df.empty:
        raise ValueError("DataFrame boş. Feature extraction yapılamaz.")

    required_columns = ["Date", "Amount", "Signed Amount", "Transaction Type", "Category"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Feature extraction için eksik kolon(lar): {missing_columns}")

    data = df.copy()

    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

    iso_calendar = data["Date"].dt.isocalendar()
    data["Year"] = iso_calendar.year.astype(int)
    data["Week"] = iso_calendar.week.astype(int)
    data["YearWeek"] = data["Year"].astype(str) + "-W" + data["Week"].astype(str).str.zfill(2)

    data["Expense"] = np.where(data["Transaction Type"] == "debit", data["Amount"], 0.0)
    data["Income"] = np.where(data["Transaction Type"] == "credit", data["Amount"], 0.0)

    weekly_base = data.groupby("YearWeek").agg(
        total_transactions=("Amount", "count"),
        total_expense=("Expense", "sum"),
        total_income=("Income", "sum"),
        avg_expense=("Expense", lambda x: x[x > 0].mean() if (x > 0).any() else 0.0),
        max_expense=("Expense", "max"),
        min_expense=("Expense", lambda x: x[x > 0].min() if (x > 0).any() else 0.0),
        std_expense=("Expense", lambda x: x[x > 0].std() if (x[x > 0].shape[0] > 1) else 0.0),
        expense_transaction_count=("Expense", lambda x: int((x > 0).sum())),
        income_transaction_count=("Income", lambda x: int((x > 0).sum())),
        active_days=("Date", lambda x: x.dt.date.nunique()),
        unique_categories=("Category", "nunique"),
        weekend_expense=("Expense", lambda x: x[data.loc[x.index, "IsWeekend"] == 1].sum()),
        weekday_expense=("Expense", lambda x: x[data.loc[x.index, "IsWeekend"] == 0].sum()),
    ).reset_index()

    fill_zero_cols = [
        "avg_expense", "max_expense", "min_expense", "std_expense",
        "weekend_expense", "weekday_expense"
    ]
    weekly_base[fill_zero_cols] = weekly_base[fill_zero_cols].fillna(0.0)

    weekly_base["net_cash_flow"] = weekly_base["total_income"] - weekly_base["total_expense"]

    weekly_base["expense_to_income_ratio"] = np.where(
        weekly_base["total_income"] > 0,
        weekly_base["total_expense"] / weekly_base["total_income"],
        0.0
    )

    weekly_base["weekend_expense_ratio"] = np.where(
        weekly_base["total_expense"] > 0,
        weekly_base["weekend_expense"] / weekly_base["total_expense"],
        0.0
    )

    weekly_base["weekday_expense_ratio"] = np.where(
        weekly_base["total_expense"] > 0,
        weekly_base["weekday_expense"] / weekly_base["total_expense"],
        0.0
    )

    weekly_base["avg_transaction_value"] = np.where(
        weekly_base["total_transactions"] > 0,
        (weekly_base["total_expense"] + weekly_base["total_income"]) / weekly_base["total_transactions"],
        0.0
    )

    weekly_base["expense_frequency_ratio"] = np.where(
        weekly_base["total_transactions"] > 0,
        weekly_base["expense_transaction_count"] / weekly_base["total_transactions"],
        0.0
    )

    weekly_base["income_presence_ratio"] = np.where(
        weekly_base["total_transactions"] > 0,
        weekly_base["income_transaction_count"] / weekly_base["total_transactions"],
        0.0
    )

    weekly_base["expense_volatility_ratio"] = np.where(
        weekly_base["avg_expense"] > 0,
        weekly_base["std_expense"] / weekly_base["avg_expense"],
        0.0
    )

    weekly_base["active_days_ratio"] = np.where(
        weekly_base["total_transactions"] > 0,
        weekly_base["active_days"] / 7.0,
        0.0
    )

    weekly_base["zero_income_week"] = (weekly_base["total_income"] <= 0).astype(int)

    category_weekly = (
        data.groupby(["YearWeek", "Category"])["Expense"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    category_columns = [col for col in category_weekly.columns if col != "YearWeek"]
    category_weekly.rename(
        columns={col: f"cat_{str(col).strip().lower().replace(' ', '_')}_amount" for col in category_columns},
        inplace=True
    )

    features_df = weekly_base.merge(category_weekly, on="YearWeek", how="left")

    category_amount_cols = [
        col for col in features_df.columns
        if col.startswith("cat_") and col.endswith("_amount")
    ]

    for col in category_amount_cols:
        ratio_col = col.replace("_amount", "_ratio")
        features_df[ratio_col] = np.where(
            features_df["total_expense"] > 0,
            features_df[col] / features_df["total_expense"],
            0.0
        )

    category_ratio_cols = [
        col for col in features_df.columns
        if col.startswith("cat_") and col.endswith("_ratio")
    ]

    if category_ratio_cols:
        ratio_matrix = features_df[category_ratio_cols].to_numpy(dtype=float)
        features_df["dominant_category_ratio"] = ratio_matrix.max(axis=1)

        entropy_values = []
        top2_sum_values = []
        for row in ratio_matrix:
            row_positive = row[row > 0]
            if len(row_positive) == 0:
                entropy_values.append(0.0)
                top2_sum_values.append(0.0)
            else:
                entropy_values.append(float(-(row_positive * np.log(row_positive + 1e-12)).sum()))
                row_sorted = np.sort(row_positive)[::-1]
                top2_sum_values.append(float(row_sorted[:2].sum()))
        features_df["category_entropy"] = entropy_values
        features_df["top2_category_ratio_sum"] = top2_sum_values
    else:
        features_df["dominant_category_ratio"] = 0.0
        features_df["category_entropy"] = 0.0
        features_df["top2_category_ratio_sum"] = 0.0

    features_df = features_df.sort_values("YearWeek").reset_index(drop=True)
    features_df["rolling_4w_expense_mean"] = features_df["total_expense"].rolling(window=4, min_periods=1).mean()
    features_df["rolling_4w_income_mean"] = features_df["total_income"].rolling(window=4, min_periods=1).mean()

    features_df["expense_vs_rolling_mean_ratio"] = np.where(
        features_df["rolling_4w_expense_mean"] > 0,
        features_df["total_expense"] / features_df["rolling_4w_expense_mean"],
        0.0
    )
    features_df["income_vs_rolling_mean_ratio"] = np.where(
        features_df["rolling_4w_income_mean"] > 0,
        features_df["total_income"] / features_df["rolling_4w_income_mean"],
        0.0
    )

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = (
        features_df[numeric_cols]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    print(f"Feature extraction tamamlandı. Oluşan haftalık profil sayısı: {len(features_df)}")
    print(f"Toplam feature sayısı: {features_df.shape[1]}")
    return features_df


def exclude_profile_outlier_weeks(
    features_df: pd.DataFrame,
    zscore_threshold: float = 3.5,
    outlier_feature_threshold: int = 3,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Profiling feature uzayında robust outlier temizliği.
    Bir hafta, seçilen önemli feature'ların en az N tanesinde robust z-score threshold'unu aşarsa çıkarılır.
    """
    print("4.) Profiling feature set üzerinde robust outlier kontrolü yapılıyor.")

    df = features_df.copy()

    candidate_cols = [
        col for col in [
            "total_expense",
            "total_income",
            "expense_to_income_ratio",
            "weekend_expense_ratio",
            "avg_transaction_value",
            "expense_frequency_ratio",
            "expense_volatility_ratio",
            "dominant_category_ratio",
            "category_entropy",
            "top2_category_ratio_sum",
            "expense_vs_rolling_mean_ratio",
        ] if col in df.columns
    ]

    if not candidate_cols:
        print("Outlier kontrolü için uygun kolon bulunamadı. Tüm haftalar kullanılacak.")
        return df, []

    robust_flags = pd.DataFrame(index=df.index)

    for col in candidate_cols:
        series = df[col].astype(float)
        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            robust_z = pd.Series(np.zeros(len(series)), index=series.index)
        else:
            robust_z = 0.6745 * (series - median) / mad

        robust_flags[col] = robust_z.abs() > zscore_threshold

    df["outlier_feature_count"] = robust_flags.sum(axis=1)
    removal_mask = df["outlier_feature_count"] >= outlier_feature_threshold

    removed_weeks = df.loc[removal_mask, "YearWeek"].tolist()
    cleaned_df = df.loc[~removal_mask].copy()

    print(f"Çıkarılan outlier hafta sayısı: {len(removed_weeks)}")
    if removed_weeks:
        print(f"Çıkarılan haftalar: {removed_weeks}")

    cleaned_df = cleaned_df.drop(columns=["outlier_feature_count"], errors="ignore")
    return cleaned_df, removed_weeks


def prepare_kmeans_input(
    features_df: pd.DataFrame,
    variance_threshold: float = 0.01,
) -> Tuple[pd.DataFrame, VarianceThreshold, StandardScaler, List[str]]:
    print("5.) KMeans input hazırlama başlıyor")

    data = features_df.copy()
    data = data.drop(columns=["YearWeek"], errors="ignore")

    numeric_data = data.select_dtypes(include=[np.number]).copy()
    if numeric_data.empty:
        raise ValueError("Numeric feature bulunamadı.")

    print(f"Başlangıç feature sayısı: {numeric_data.shape[1]}")

    selector = VarianceThreshold(threshold=variance_threshold)
    filtered_array = selector.fit_transform(numeric_data)
    selected_columns = numeric_data.columns[selector.get_support()].tolist()

    filtered_df = pd.DataFrame(filtered_array, columns=selected_columns, index=numeric_data.index)
    print(f"Low-variance sonrası feature sayısı: {filtered_df.shape[1]}")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(filtered_df)
    scaled_df = pd.DataFrame(scaled_array, columns=selected_columns, index=filtered_df.index)

    scaled_df = scaled_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    print(f"KMeans input hazır. Shape: {scaled_df.shape}")
    return scaled_df, selector, scaler, selected_columns


def _calculate_balance_score(cluster_counts: pd.Series) -> float:
    proportions = cluster_counts / cluster_counts.sum()
    ideal = 1.0 / len(cluster_counts)
    imbalance = np.abs(proportions - ideal).sum()
    return float(max(0.0, 1.0 - imbalance))


def _is_small_cluster(cluster_size: int, total_count: int, min_cluster_ratio: float) -> bool:
    return (cluster_size / total_count) < min_cluster_ratio


def _generate_profile_label(row: pd.Series, total_count: int, min_cluster_ratio: float) -> str:
    cluster_size = int(row.get("cluster_size", 0))
    if _is_small_cluster(cluster_size, total_count, min_cluster_ratio):
        return "uc_davranis_grubu"

    expense_ratio = row.get("expense_to_income_ratio_mean", 0.0)
    weekend_ratio = row.get("weekend_expense_ratio_mean", 0.0)
    volatility = row.get("expense_volatility_ratio_mean", 0.0)
    dominant_cat = row.get("dominant_category_ratio_mean", 0.0)
    top2_cat = row.get("top2_category_ratio_sum_mean", 0.0)
    total_income = row.get("total_income_mean", 0.0)
    active_days_ratio = row.get("active_days_ratio_mean", 0.0)
    expense_vs_roll = row.get("expense_vs_rolling_mean_ratio_mean", 0.0)
    zero_income_week = row.get("zero_income_week_mean", 0.0)
    category_entropy = row.get("category_entropy_mean", 0.0)

    if expense_ratio < 0.45 and volatility < 0.50 and expense_vs_roll <= 1.05:
        return "tasarruf_odakli"

    if dominant_cat > 0.55 or (top2_cat > 0.75 and category_entropy < 1.05):
        return "kategori_yogun_harcama"

    if weekend_ratio > 0.42:
        return "hafta_sonu_agirlikli"

    if expense_ratio > 0.90 and volatility > 0.70 and expense_vs_roll > 1.10:
        return "dengesiz_yuksek_harcama"

    if zero_income_week > 0.30:
        return "gelirsiz_harcama_donemi"

    if total_income > 0 and expense_ratio <= 0.75 and active_days_ratio >= 0.35:
        return "dengeli_harcama"

    return "genel_davranis_profili"


def _build_cluster_summary(
    clustered_features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    min_cluster_ratio: float,
) -> pd.DataFrame:
    temp_df = clustered_features_df.copy()
    temp_df["cluster_label"] = cluster_labels

    summary_cols = [
        col for col in [
            "total_expense",
            "total_income",
            "net_cash_flow",
            "expense_to_income_ratio",
            "weekend_expense_ratio",
            "weekday_expense_ratio",
            "avg_transaction_value",
            "expense_frequency_ratio",
            "income_presence_ratio",
            "expense_volatility_ratio",
            "dominant_category_ratio",
            "top2_category_ratio_sum",
            "category_entropy",
            "unique_categories",
            "active_days_ratio",
            "zero_income_week",
            "expense_vs_rolling_mean_ratio",
            "income_vs_rolling_mean_ratio",
        ] if col in temp_df.columns
    ]

    cluster_summary = temp_df.groupby("cluster_label")[summary_cols].agg(["mean", "median"])
    cluster_summary.columns = [f"{col}_{agg}" for col, agg in cluster_summary.columns.to_flat_index()]
    cluster_summary = cluster_summary.reset_index()

    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_sizes = cluster_counts.rename("cluster_size").reset_index()
    cluster_sizes.columns = ["cluster_label", "cluster_size"]

    cluster_summary = cluster_summary.merge(cluster_sizes, on="cluster_label", how="left")

    total_count = int(cluster_counts.sum())
    cluster_summary["profile_label"] = cluster_summary.apply(
        lambda row: _generate_profile_label(row, total_count, min_cluster_ratio),
        axis=1
    )

    return cluster_summary


def train_weekly_profiler_model(
    X: pd.DataFrame,
    features_df: pd.DataFrame,
    selector: VarianceThreshold,
    scaler: StandardScaler,
    selected_feature_names: List[str],
    config: ProfilingConfig,
):
    print("6.) Weekly profiling KMeans model eğitimi başlıyor")

    n_samples = len(X)
    valid_candidate_ks = tuple(k for k in config.candidate_ks if k < n_samples)
    if not valid_candidate_ks:
        raise ValueError(f"Geçerli k değeri kalmadı. n_samples={n_samples}, candidate_ks={config.candidate_ks}")

    eval_rows = []
    fitted_models = {}

    print(f"Denenecek k değerleri: {list(valid_candidate_ks)}")

    for k in valid_candidate_ks:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
            algorithm="lloyd",
        )

        labels = model.fit_predict(X)
        counts = pd.Series(labels).value_counts().sort_index()

        silhouette = float(silhouette_score(X, labels))
        db_score = float(davies_bouldin_score(X, labels))
        balance_score = float(_calculate_balance_score(counts))
        min_cluster_fraction = float(counts.min() / counts.sum())

        composite_score = (
            (0.40 * silhouette) +
            (0.30 * balance_score) +
            (0.20 * min_cluster_fraction) -
            (0.10 * db_score)
        )

        if min_cluster_fraction < config.min_cluster_ratio:
            composite_score -= 0.25

        eval_rows.append({
            "k": int(k),
            "silhouette": silhouette,
            "davies_bouldin": db_score,
            "balance_score": balance_score,
            "min_cluster_fraction": min_cluster_fraction,
            "composite_score": composite_score,
            "cluster_counts": counts.to_dict(),
        })
        fitted_models[k] = model

        print(
            f"k={k} | silhouette={silhouette:.4f} | "
            f"db={db_score:.4f} | balance={balance_score:.4f} | "
            f"min_cluster_fraction={min_cluster_fraction:.4f} | "
            f"composite={composite_score:.4f} | counts={counts.to_dict()}"
        )

    evaluation_df = pd.DataFrame(eval_rows).sort_values(
        by=["composite_score", "balance_score", "silhouette"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_k = int(evaluation_df.iloc[0]["k"])
    final_model = fitted_models[best_k]
    final_labels = final_model.predict(X)

    final_counts = pd.Series(final_labels).value_counts().sort_index()
    final_silhouette = float(silhouette_score(X, final_labels))
    final_db = float(davies_bouldin_score(X, final_labels))
    final_balance = float(_calculate_balance_score(final_counts))

    print("\nModel seçim özeti:")
    print(f"Seçilen final k      : {best_k}")
    print(f"Final silhouette     : {final_silhouette:.4f}")
    print(f"Final DB score       : {final_db:.4f}")
    print(f"Final balance score  : {final_balance:.4f}")
    print(f"Cluster dağılımı     : {final_counts.to_dict()}")

    cluster_summary = _build_cluster_summary(
        clustered_features_df=features_df,
        cluster_labels=final_labels,
        min_cluster_ratio=config.min_cluster_ratio,
    )

    model_artifact = {
        "model_type": "weekly_spending_profiler",
        "model_version": "3.0.0",
        "model": final_model,
        "selector": selector,
        "scaler": scaler,
        "selected_feature_names": selected_feature_names,
        "best_k": best_k,
        "train_shape": X.shape,
        "cluster_counts": final_counts.to_dict(),
        "metrics": {
            "candidate_ks": list(valid_candidate_ks),
            "evaluation_table": evaluation_df.to_dict(orient="records"),
            "selected_k": best_k,
            "selected_model_silhouette": final_silhouette,
            "selected_model_davies_bouldin": final_db,
            "selected_model_balance_score": final_balance,
        },
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "config": {
            "variance_threshold": config.variance_threshold,
            "candidate_ks": list(config.candidate_ks),
            "min_cluster_ratio": config.min_cluster_ratio,
            "outlier_exclusion_enabled": config.outlier_exclusion_enabled,
            "outlier_feature_threshold": config.outlier_feature_threshold,
            "outlier_zscore_threshold": config.outlier_zscore_threshold,
            "n_init": config.n_init,
            "max_iter": config.max_iter,
            "random_state": config.random_state,
        },
    }

    return model_artifact, final_labels, evaluation_df, cluster_summary


if __name__ == "__main__":
    try:
        config = ProfilingConfig()

        raw_data = load_and_clean_data(DATA_PATH)
        weekly_features = extract_weekly_features(raw_data)

        original_weekly_count = len(weekly_features)
        removed_outlier_weeks = []

        if config.outlier_exclusion_enabled:
            weekly_features_for_training, removed_outlier_weeks = exclude_profile_outlier_weeks(
                weekly_features,
                zscore_threshold=config.outlier_zscore_threshold,
                outlier_feature_threshold=config.outlier_feature_threshold,
            )
        else:
            weekly_features_for_training = weekly_features.copy()

        print(f"Profiling eğitim veri boyutu: {len(weekly_features_for_training)} / {original_weekly_count}")

        X, selector, scaler, selected_feature_names = prepare_kmeans_input(
            weekly_features_for_training,
            variance_threshold=config.variance_threshold,
        )

        model_artifact, final_labels, evaluation_df, cluster_summary = train_weekly_profiler_model(
            X=X,
            features_df=weekly_features_for_training,
            selector=selector,
            scaler=scaler,
            selected_feature_names=selected_feature_names,
            config=config,
        )

        model_artifact["training_scope"] = {
            "original_weekly_count": int(original_weekly_count),
            "removed_outlier_weeks": removed_outlier_weeks,
            "final_training_weekly_count": int(len(weekly_features_for_training)),
        }

        os.makedirs(os.path.dirname(MODEL_STORE_PATH), exist_ok=True)
        joblib.dump(model_artifact, MODEL_STORE_PATH)

        print(f"\nModel başarıyla kaydedildi: {MODEL_STORE_PATH}")
        print("\nEvaluation table:")
        print(evaluation_df)

        print("\nCluster summary:")
        print(cluster_summary)

    except Exception as e:
        print(f"HATA: Profiling model eğitimi sırasında bir sorun oluştu: {str(e)}")