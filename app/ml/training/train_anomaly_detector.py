import os
import joblib
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "../data/transactions.csv")
MODEL_STORE_PATH = os.path.join(BASE_DIR, "ml/models_store/weekly_spending_anomaly_detector.joblib")


RANDOM_STATE = 42
LOW_VARIANCE_THRESHOLD = 0.01
DEFAULT_N_CLUSTERS = 2
DEFAULT_N_INIT = 20
DEFAULT_MAX_ITER = 500

# Eğitim datası üzerinden threshold üretirken:
# p95 = anomaly
# p99 = severe anomaly gibi yorumlayabilirsin
ANOMALY_PERCENTILE = 95
SEVERE_ANOMALY_PERCENTILE = 99


@dataclass(frozen=True)
class TrainingConfig:
    n_clusters: int = DEFAULT_N_CLUSTERS
    variance_threshold: float = LOW_VARIANCE_THRESHOLD
    anomaly_percentile: float = ANOMALY_PERCENTILE
    severe_anomaly_percentile: float = SEVERE_ANOMALY_PERCENTILE
    random_state: int = RANDOM_STATE
    n_init: int = DEFAULT_N_INIT
    max_iter: int = DEFAULT_MAX_ITER


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print("1.) Veri yükleniyor.")
    df = pd.read_csv(file_path)
    print(f"Yükleme tamamlandı. Kayıt sayısı: {len(df)}")

    print("2.) Veri temizleniyor.")
    required_columns = [
        "Date",
        "Description",
        "Amount",
        "Transaction Type",
        "Category",
        "Account Name",
    ]

    df.columns = df.columns.str.strip()

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik kolon(lar) bulundu: {missing_columns}")

    df = df[required_columns].copy()

    text_columns = ["Description", "Transaction Type", "Category", "Account Name"]
    for col in text_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df = df.dropna(subset=["Date", "Amount", "Transaction Type", "Category"]).copy()

    df["Transaction Type"] = df["Transaction Type"].str.lower()
    valid_transaction_types = {"debit", "credit"}
    df = df[df["Transaction Type"].isin(valid_transaction_types)].copy()

    df["Amount"] = df["Amount"].abs()
    df = df[df["Amount"] > 0].copy()

    df["Signed Amount"] = np.where(
        df["Transaction Type"] == "debit",
        -df["Amount"],
        df["Amount"]
    )

    df["Description"] = df["Description"].str.lower()
    df["Category"] = df["Category"].str.lower()
    df["Account Name"] = df["Account Name"].str.lower()

    df = df.drop_duplicates(
        subset=[
            "Date",
            "Description",
            "Amount",
            "Transaction Type",
            "Category",
            "Account Name",
        ]
    ).copy()

    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Temizleme tamamlandı. Kayıt sayısı: {len(df)}")
    return df


def extract_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    print("3.) Weekly Feature Engineering başlıyor.")

    if df.empty:
        raise ValueError("DataFrame boş. Feature extraction yapılamaz.")

    required_columns = [
        "Date",
        "Amount",
        "Signed Amount",
        "Transaction Type",
        "Category",
    ]

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
        expense_transaction_count=("Expense", lambda x: (x > 0).sum()),
        income_transaction_count=("Income", lambda x: (x > 0).sum()),
        weekend_expense=("Expense", lambda x: x[data.loc[x.index, "IsWeekend"] == 1].sum()),
        weekday_expense=("Expense", lambda x: x[data.loc[x.index, "IsWeekend"] == 0].sum()),
    ).reset_index()

    fill_zero_cols = [
        "avg_expense",
        "max_expense",
        "min_expense",
        "std_expense",
        "weekend_expense",
        "weekday_expense",
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

    category_weekly = (
        data.groupby(["YearWeek", "Category"])["Expense"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    category_columns = [col for col in category_weekly.columns if col != "YearWeek"]
    category_weekly.rename(
        columns={
            col: f"cat_{str(col).strip().lower().replace(' ', '_')}_amount"
            for col in category_columns
        },
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

    features_df["avg_transaction_value"] = np.where(
        features_df["total_transactions"] > 0,
        (features_df["total_expense"] + features_df["total_income"]) / features_df["total_transactions"],
        0.0
    )

    features_df["expense_frequency_ratio"] = np.where(
        features_df["total_transactions"] > 0,
        features_df["expense_transaction_count"] / features_df["total_transactions"],
        0.0
    )

    features_df = features_df.sort_values("YearWeek").reset_index(drop=True)

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = (
        features_df[numeric_cols]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    print(f"Feature extraction tamamlandı. Oluşan haftalık profil sayısı: {len(features_df)}")
    print(f"Toplam feature sayısı: {features_df.shape[1]}")
    return features_df


def prepare_anomaly_input(
    features_df: pd.DataFrame,
    variance_threshold: float,
) -> Tuple[pd.DataFrame, VarianceThreshold, StandardScaler, List[str]]:
    print("4.) Anomaly input hazırlama başlıyor")

    if features_df.empty:
        raise ValueError("Feature DataFrame boş.")

    data = features_df.copy()

    protected_columns = ["YearWeek"]
    feature_data = data.drop(columns=[col for col in protected_columns if col in data.columns])

    numeric_data = feature_data.select_dtypes(include=[np.number]).copy()
    if numeric_data.shape[1] == 0:
        raise ValueError("Numeric feature bulunamadı.")

    print(f"Başlangıç feature sayısı: {numeric_data.shape[1]}")

    selector = VarianceThreshold(threshold=variance_threshold)
    filtered_array = selector.fit_transform(numeric_data)
    selected_feature_names = numeric_data.columns[selector.get_support()].tolist()

    filtered_df = pd.DataFrame(
        filtered_array,
        columns=selected_feature_names,
        index=numeric_data.index,
    )

    print(f"Low-variance sonrası feature sayısı: {filtered_df.shape[1]}")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(filtered_df)

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=selected_feature_names,
        index=filtered_df.index,
    )

    scaled_df = scaled_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    print(f"Anomaly input hazır. Shape: {scaled_df.shape}")

    return scaled_df, selector, scaler, selected_feature_names


def compute_anomaly_scores(
    X: pd.DataFrame,
    model: KMeans,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    nearest_distance:
        Her sample'ın en yakın cluster merkezine uzaklığı.
        Anomaly score olarak kullanılacak.

    full_distances:
        Tüm cluster merkezlerine olan uzaklıklar.
    """
    full_distances = model.transform(X)
    nearest_distance = np.min(full_distances, axis=1)
    return nearest_distance, full_distances


def extract_top_feature_contributions(
    original_scaled_row: pd.Series,
    centroid_scaled_row: pd.Series,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    abs_diff = (original_scaled_row - centroid_scaled_row).abs().sort_values(ascending=False)

    contributions = []
    for feature_name, diff_value in abs_diff.head(top_n).items():
        contributions.append(
            {
                "feature": feature_name,
                "absolute_deviation": float(diff_value),
                "sample_value_scaled": float(original_scaled_row[feature_name]),
                "centroid_value_scaled": float(centroid_scaled_row[feature_name]),
            }
        )
    return contributions


def train_weekly_kmeans_anomaly_detector(
    X: pd.DataFrame,
    selector: VarianceThreshold,
    scaler: StandardScaler,
    selected_feature_names: List[str],
    config: TrainingConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    print("5.) Weekly KMeans anomaly detector eğitimi başlıyor")

    if X is None or len(X) == 0:
        raise ValueError("Eğitim verisi boş.")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=selected_feature_names)

    n_samples, n_features = X.shape
    print(f"Eğitim verisi shape: {X.shape}")

    if n_samples < config.n_clusters:
        raise ValueError(
            f"n_clusters={config.n_clusters} için yetersiz örnek sayısı: {n_samples}"
        )

    model = KMeans(
        n_clusters=config.n_clusters,
        init="k-means++",
        n_init=config.n_init,
        max_iter=config.max_iter,
        random_state=config.random_state,
        algorithm="lloyd",
    )

    cluster_labels = model.fit_predict(X)

    anomaly_scores, full_distances = compute_anomaly_scores(X, model)

    anomaly_threshold = float(np.percentile(anomaly_scores, config.anomaly_percentile))
    severe_anomaly_threshold = float(np.percentile(anomaly_scores, config.severe_anomaly_percentile))

    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index().to_dict()

    train_result_df = X.copy()
    train_result_df["cluster_label"] = cluster_labels
    train_result_df["anomaly_score"] = anomaly_scores
    train_result_df["is_anomaly"] = train_result_df["anomaly_score"] > anomaly_threshold
    train_result_df["is_severe_anomaly"] = train_result_df["anomaly_score"] > severe_anomaly_threshold

    print("\nModel özeti:")
    print(f"n_clusters               : {config.n_clusters}")
    print(f"anomaly_threshold(p{config.anomaly_percentile})       : {anomaly_threshold:.6f}")
    print(f"severe_threshold(p{config.severe_anomaly_percentile}) : {severe_anomaly_threshold:.6f}")
    print(f"cluster dağılımı         : {cluster_counts}")
    print(f"anomaly sayısı           : {int(train_result_df['is_anomaly'].sum())}")
    print(f"severe anomaly sayısı    : {int(train_result_df['is_severe_anomaly'].sum())}")

    model_artifact = {
        "model_type": "weekly_spending_kmeans_anomaly_detector",
        "model_version": "1.0.0",
        "model": model,
        "selector": selector,
        "scaler": scaler,
        "selected_feature_names": selected_feature_names,
        "training_shape": X.shape,
        "cluster_counts": cluster_counts,
        "thresholds": {
            "anomaly_score_threshold": anomaly_threshold,
            "severe_anomaly_score_threshold": severe_anomaly_threshold,
            "anomaly_percentile": config.anomaly_percentile,
            "severe_anomaly_percentile": config.severe_anomaly_percentile,
        },
        "training_metrics": {
            "score_min": float(np.min(anomaly_scores)),
            "score_mean": float(np.mean(anomaly_scores)),
            "score_median": float(np.median(anomaly_scores)),
            "score_max": float(np.max(anomaly_scores)),
        },
        "config": {
            "n_clusters": config.n_clusters,
            "variance_threshold": config.variance_threshold,
            "random_state": config.random_state,
            "n_init": config.n_init,
            "max_iter": config.max_iter,
        },
    }

    os.makedirs(os.path.dirname(MODEL_STORE_PATH), exist_ok=True)
    joblib.dump(model_artifact, MODEL_STORE_PATH)
    print(f"Model başarıyla kaydedildi: {MODEL_STORE_PATH}")

    return model_artifact, train_result_df


def score_weekly_anomalies(
    features_df: pd.DataFrame,
    artifact: Dict[str, Any],
    top_n_features: int = 5,
) -> pd.DataFrame:
    """
    Eğitilmiş artifact ile yeni haftalık feature'ları skorlar.
    """
    if features_df.empty:
        raise ValueError("Scoring için features_df boş.")

    required_keys = [
        "model",
        "selector",
        "scaler",
        "selected_feature_names",
        "thresholds",
    ]
    missing_keys = [key for key in required_keys if key not in artifact]
    if missing_keys:
        raise ValueError(f"Artifact eksik anahtar(lar): {missing_keys}")

    model: KMeans = artifact["model"]
    selector: VarianceThreshold = artifact["selector"]
    scaler: StandardScaler = artifact["scaler"]
    selected_feature_names: List[str] = artifact["selected_feature_names"]

    thresholds = artifact["thresholds"]
    anomaly_threshold = thresholds["anomaly_score_threshold"]
    severe_anomaly_threshold = thresholds["severe_anomaly_score_threshold"]

    output_df = features_df.copy()

    numeric_data = output_df.select_dtypes(include=[np.number]).copy()
    if numeric_data.empty:
        raise ValueError("Scoring için numeric feature bulunamadı.")

    filtered_array = selector.transform(numeric_data)
    filtered_df = pd.DataFrame(
        filtered_array,
        columns=selected_feature_names,
        index=output_df.index,
    )

    scaled_array = scaler.transform(filtered_df)
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=selected_feature_names,
        index=output_df.index,
    )

    scaled_df = scaled_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    cluster_labels = model.predict(scaled_df)
    full_distances = model.transform(scaled_df)
    anomaly_scores = np.min(full_distances, axis=1)
    nearest_cluster_indices = np.argmin(full_distances, axis=1)

    output_df["cluster_label"] = cluster_labels
    output_df["anomaly_score"] = anomaly_scores
    output_df["is_anomaly"] = output_df["anomaly_score"] > anomaly_threshold
    output_df["is_severe_anomaly"] = output_df["anomaly_score"] > severe_anomaly_threshold

    explanations: List[List[Dict[str, Any]]] = []
    centroids = pd.DataFrame(model.cluster_centers_, columns=selected_feature_names)

    for row_idx in scaled_df.index:
        assigned_cluster = int(nearest_cluster_indices[list(scaled_df.index).index(row_idx)])
        row_scaled = scaled_df.loc[row_idx]
        centroid_scaled = centroids.loc[assigned_cluster]

        top_contributions = extract_top_feature_contributions(
            original_scaled_row=row_scaled,
            centroid_scaled_row=centroid_scaled,
            top_n=top_n_features,
        )
        explanations.append(top_contributions)

    output_df["top_feature_contributions"] = explanations
    return output_df


if __name__ == "__main__":
    try:
        config = TrainingConfig(
            n_clusters=2,
            variance_threshold=0.01,
            anomaly_percentile=95,
            severe_anomaly_percentile=99,
            random_state=42,
            n_init=20,
            max_iter=500,
        )

        raw_data = load_and_clean_data(DATA_PATH)
        feature_df = extract_weekly_features(raw_data)

        X, selector, scaler, selected_feature_names = prepare_anomaly_input(
            feature_df,
            variance_threshold=config.variance_threshold,
        )

        artifact, train_scores_df = train_weekly_kmeans_anomaly_detector(
            X=X,
            selector=selector,
            scaler=scaler,
            selected_feature_names=selected_feature_names,
            config=config,
        )

        # Eğitim datası ile explainable scoring örneği
        scored_feature_df = score_weekly_anomalies(
            features_df=feature_df,
            artifact=artifact,
            top_n_features=5,
        )

        print("\n6.) Örnek skorlanmış sonuçlar:")
        display_cols = [
            col for col in [
                "YearWeek",
                "total_expense",
                "total_income",
                "net_cash_flow",
                "cluster_label",
                "anomaly_score",
                "is_anomaly",
                "is_severe_anomaly",
            ] if col in scored_feature_df.columns
        ]

        print(scored_feature_df[display_cols].sort_values("anomaly_score", ascending=False).head(10))
        print("\nModel eğitimi ve anomaly scoring tamamlandı.")

    except Exception as e:
        print(f"HATA: İşlem sırasında bir sorun oluştu: {str(e)}")