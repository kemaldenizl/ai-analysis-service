import os
import sys
import math
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ml.training.train_profiler import (
    load_and_clean_data as load_profile_data,
    extract_weekly_features as extract_profile_weekly_features,
)

from app.ml.training.train_anomaly_detector import (
    load_and_clean_data as load_anomaly_data,
    extract_weekly_features as extract_anomaly_weekly_features,
    score_weekly_anomalies,
)

from app.ml.training.train_forecaster import (
    load_and_clean_data as load_forecast_data,
    _build_monthly_base_table,
    _build_month_snapshot_row,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "../data/transactions.csv")

PROFILER_MODEL_PATH = os.path.join(BASE_DIR, "ml/models_store/spending_profiler.joblib")
ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "ml/models_store/weekly_spending_anomaly_detector.joblib")
FORECAST_MODEL_PATH = os.path.join(BASE_DIR, "ml/models_store/monthly_forecaster.joblib")


@dataclass(frozen=True)
class FinancialRiskConfig:
    profile_weight: float = 0.35
    anomaly_weight: float = 0.25
    forecast_weight: float = 0.40

    low_threshold: float = 40.0
    medium_threshold: float = 70.0

    sigmoid_sharpness: float = 6.0
    min_budget_for_risk: float = 1.0


class FinancialRiskPredictor:
    """
    Tek sorumluluk:
    - Profil sonucu
    - Anomali sonucu
    - Forecast sonucu / budget overrun ihtimali
    birleşiminden 0-100 arası explainable risk skoru üretmek.

    Beklenen kullanım:
        predictor = FinancialRiskPredictor()
        result = predictor.predict_from_csv(file_path=".../transactions.csv", budget=3500.0)

    veya
        result = predictor.predict(df=my_dataframe, budget=3500.0)
    """

    def __init__(
        self,
        profiler_model_path: str = PROFILER_MODEL_PATH,
        anomaly_model_path: str = ANOMALY_MODEL_PATH,
        forecast_model_path: str = FORECAST_MODEL_PATH,
        config: FinancialRiskConfig = FinancialRiskConfig(),
    ) -> None:
        self.config = config
        self.profiler_artifact = self._load_artifact(profiler_model_path, "profiler")
        self.anomaly_artifact = self._load_artifact(anomaly_model_path, "anomaly")
        self.forecast_artifact = self._load_artifact(forecast_model_path, "forecast")

    @staticmethod
    def _load_artifact(path: str, name: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} model artifact bulunamadı: {path}")
        return joblib.load(path)

    def predict_from_csv(self, file_path: str = DEFAULT_DATA_PATH, budget: float = 0.0) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")

        df = pd.read_csv(file_path)
        return self.predict(df=df, budget=budget)

    def predict(self, df: pd.DataFrame, budget: float) -> Dict[str, Any]:
        if df is None or df.empty:
            raise ValueError("Risk skoru hesaplamak için boş olmayan bir DataFrame gerekli.")

        if budget is None:
            budget = 0.0

        profile_result = self._predict_profile(df)
        anomaly_result = self._predict_anomaly(df)
        forecast_result = self._predict_forecast(df, budget=budget)

        profile_norm = profile_result["profile_risk_norm"]
        anomaly_norm = anomaly_result["anomaly_risk_norm"]
        forecast_norm = forecast_result["forecast_risk_norm"]

        risk_score = round(
            (
                self.config.profile_weight * profile_norm +
                self.config.anomaly_weight * anomaly_norm +
                self.config.forecast_weight * forecast_norm
            ) * 100.0,
            2,
        )

        risk_level = self._classify_risk_level(risk_score)
        dominant_driver = self._detect_dominant_driver(
            profile_norm=profile_norm,
            anomaly_norm=anomaly_norm,
            forecast_norm=forecast_norm,
        )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "dominant_driver": dominant_driver,
            "weights": {
                "profile_weight": self.config.profile_weight,
                "anomaly_weight": self.config.anomaly_weight,
                "forecast_weight": self.config.forecast_weight,
            },
            "components": {
                "profile": profile_result,
                "anomaly": anomaly_result,
                "forecast": forecast_result,
            },
            "explanation": self._build_explanation(
                risk_level=risk_level,
                profile_result=profile_result,
                anomaly_result=anomaly_result,
                forecast_result=forecast_result,
            ),
        }

    def _predict_profile(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        clean_df = load_profile_data_from_df(raw_df)
        weekly_features = extract_profile_weekly_features(clean_df)

        if weekly_features.empty:
            raise ValueError("Profil skoru için haftalık feature üretilemedi.")

        latest_week_row = weekly_features.iloc[[-1]].copy()

        selector = self.profiler_artifact["selector"]
        scaler = self.profiler_artifact["scaler"]
        model = self.profiler_artifact["model"]
        selected_feature_names = self.profiler_artifact["selected_feature_names"]

        numeric_data = latest_week_row.drop(columns=["YearWeek"], errors="ignore").select_dtypes(include=[np.number]).copy()
        filtered_array = selector.transform(numeric_data)
        filtered_df = pd.DataFrame(filtered_array, columns=selected_feature_names, index=latest_week_row.index)
        scaled_array = scaler.transform(filtered_df)
        scaled_df = pd.DataFrame(scaled_array, columns=selected_feature_names, index=latest_week_row.index)
        scaled_df = scaled_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        cluster_id = int(model.predict(scaled_df)[0])
        latest_year_week = str(latest_week_row["YearWeek"].iloc[0])

        cluster_summary = self.profiler_artifact.get("cluster_summary", [])
        profile_label = self._find_profile_label(cluster_summary, cluster_id)
        cluster_risk_map = self._build_cluster_risk_map(cluster_summary)
        profile_risk_norm = float(cluster_risk_map.get(cluster_id, 0.50))

        return {
            "year_week": latest_year_week,
            "cluster_id": cluster_id,
            "profile_label": profile_label,
            "profile_risk_norm": round(profile_risk_norm, 4),
            "cluster_risk_map": {str(k): round(v, 4) for k, v in cluster_risk_map.items()},
        }

    def _predict_anomaly(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        clean_df = load_anomaly_data_from_df(raw_df)
        weekly_features = extract_anomaly_weekly_features(clean_df)

        if weekly_features.empty:
            raise ValueError("Anomali skoru için haftalık feature üretilemedi.")

        scored_df = score_weekly_anomalies(
            features_df=weekly_features,
            artifact=self.anomaly_artifact,
            top_n_features=5,
        )

        latest_row = scored_df.iloc[-1]
        anomaly_score = float(latest_row["anomaly_score"])

        thresholds = self.anomaly_artifact.get("thresholds", {})
        anomaly_threshold = float(thresholds.get("anomaly_score_threshold", 0.0))
        severe_threshold = float(
            thresholds.get("severe_anomaly_score_threshold", max(anomaly_threshold, 1e-9))
        )

        anomaly_risk_norm = self._normalize_anomaly_score(
            anomaly_score=anomaly_score,
            anomaly_threshold=anomaly_threshold,
            severe_threshold=severe_threshold,
        )

        return {
            "year_week": str(latest_row.get("YearWeek", "")),
            "anomaly_score": round(anomaly_score, 6),
            "anomaly_threshold": round(anomaly_threshold, 6),
            "severe_anomaly_threshold": round(severe_threshold, 6),
            "is_anomaly": bool(latest_row.get("is_anomaly", False)),
            "is_severe_anomaly": bool(latest_row.get("is_severe_anomaly", False)),
            "anomaly_risk_norm": round(anomaly_risk_norm, 4),
            "top_feature_contributions": latest_row.get("top_feature_contributions", []),
        }

    def _predict_forecast(self, raw_df: pd.DataFrame, budget: float) -> Dict[str, Any]:
        clean_df = load_forecast_data_from_df(raw_df)
        if clean_df.empty:
            raise ValueError("Forecast için temiz veri üretilemedi.")

        forecast_model = self.forecast_artifact["model"]
        feature_columns = self.forecast_artifact["feature_columns"]
        snapshot_day = int(self.forecast_artifact.get("snapshot_day", 15))

        latest_month = str(clean_df["YearMonth"].max())
        current_month_df = clean_df.loc[clean_df["YearMonth"] == latest_month].copy()
        if current_month_df.empty:
            raise ValueError("Forecast için son ay verisi bulunamadı.")

        latest_available_day = int(current_month_df["Day"].max())
        effective_snapshot_day = min(snapshot_day, latest_available_day)

        monthly_base = _build_monthly_base_table(clean_df)

        snapshot_row = _build_month_snapshot_row(
            month_df=current_month_df,
            monthly_base_lookup=monthly_base,
            snapshot_day=effective_snapshot_day,
        )

        if not snapshot_row:
            raise ValueError("Forecast snapshot row üretilemedi.")

        snapshot_df = pd.DataFrame([snapshot_row])

        for col in feature_columns:
            if col not in snapshot_df.columns:
                snapshot_df[col] = np.nan

        X_pred = snapshot_df[feature_columns].copy()
        predicted_month_end_expense = float(forecast_model.predict(X_pred)[0])

        spent_so_far = float(snapshot_df["total_expense_so_far"].iloc[0])
        budget_value = float(max(budget, 0.0))

        if budget_value >= self.config.min_budget_for_risk:
            budget_usage_ratio = predicted_month_end_expense / budget_value
            budget_remaining = budget_value - spent_so_far
            expected_overrun_amount = max(0.0, predicted_month_end_expense - budget_value)
            budget_overrun_probability = self._sigmoid(
                (budget_usage_ratio - 1.0) * self.config.sigmoid_sharpness
            )
            forecast_risk_norm = min(1.0, max(0.0, budget_overrun_probability))
        else:
            budget_usage_ratio = 0.0
            budget_remaining = 0.0
            expected_overrun_amount = 0.0
            budget_overrun_probability = 0.0

            # budget verilmediyse yalnızca tempo bazlı güvenli fallback
            prev_mean = float(snapshot_df.get("rolling_3m_expense_mean", pd.Series([0.0])).iloc[0])
            if prev_mean > 0:
                forecast_risk_norm = min(1.0, max(0.0, predicted_month_end_expense / prev_mean))
            else:
                forecast_risk_norm = 0.5

        return {
            "year_month": latest_month,
            "model_snapshot_day": snapshot_day,
            "effective_snapshot_day": effective_snapshot_day,
            "spent_so_far": round(spent_so_far, 2),
            "predicted_month_end_expense": round(predicted_month_end_expense, 2),
            "budget": round(budget_value, 2),
            "budget_remaining": round(budget_remaining, 2),
            "budget_usage_ratio": round(budget_usage_ratio, 4),
            "expected_overrun_amount": round(expected_overrun_amount, 2),
            "budget_overrun_probability": round(budget_overrun_probability, 4),
            "forecast_risk_norm": round(float(forecast_risk_norm), 4),
        }

    def _build_cluster_risk_map(self, cluster_summary: List[Dict[str, Any]]) -> Dict[int, float]:
        """
        Training tarafında ayrıca risk map kaydetmeye gerek bırakmadan
        cluster summary üzerinden dinamik risk map üretir.

        Kullanılan sinyaller:
        - expense_to_income_ratio_mean
        - expense_volatility_ratio_mean
        - dominant_category_ratio_mean
        - weekend_expense_ratio_mean
        - zero_income_week_mean
        - expense_vs_rolling_mean_ratio_mean
        """
        if not cluster_summary:
            return {}

        rows = []
        for row in cluster_summary:
            cluster_id = int(row["cluster_label"])

            raw_score = (
                0.28 * float(row.get("expense_to_income_ratio_mean", 0.0)) +
                0.20 * float(row.get("expense_volatility_ratio_mean", 0.0)) +
                0.14 * float(row.get("dominant_category_ratio_mean", 0.0)) +
                0.10 * float(row.get("weekend_expense_ratio_mean", 0.0)) +
                0.18 * float(row.get("zero_income_week_mean", 0.0)) +
                0.10 * max(0.0, float(row.get("expense_vs_rolling_mean_ratio_mean", 0.0)) - 1.0)
            )

            rows.append((cluster_id, raw_score))

        raw_values = np.array([x[1] for x in rows], dtype=float)
        if len(raw_values) == 1:
            return {rows[0][0]: 0.50}

        min_v = float(np.min(raw_values))
        max_v = float(np.max(raw_values))

        risk_map: Dict[int, float] = {}
        for cluster_id, raw_score in rows:
            if math.isclose(max_v, min_v):
                normalized = 0.50
            else:
                normalized = (raw_score - min_v) / (max_v - min_v)

            # aşırı 0 / 1 sertliği yerine biraz sıkıştır
            normalized = 0.15 + (0.85 * normalized)
            risk_map[cluster_id] = float(min(1.0, max(0.0, normalized)))

        return risk_map

    @staticmethod
    def _find_profile_label(cluster_summary: List[Dict[str, Any]], cluster_id: int) -> str:
        for row in cluster_summary:
            if int(row.get("cluster_label", -1)) == cluster_id:
                return str(row.get("profile_label", "genel_davranis_profili"))
        return "genel_davranis_profili"

    @staticmethod
    def _normalize_anomaly_score(anomaly_score: float, anomaly_threshold: float, severe_threshold: float) -> float:
        """
        Explainable piecewise normalize:
        - threshold altı: düşük risk bandı
        - threshold-severe arası: hızlı artan risk
        - severe üstü: maksimum risk
        """
        anomaly_threshold = max(anomaly_threshold, 1e-9)
        severe_threshold = max(severe_threshold, anomaly_threshold + 1e-9)

        if anomaly_score <= 0:
            return 0.0

        if anomaly_score < anomaly_threshold:
            return min(0.69, anomaly_score / anomaly_threshold * 0.69)

        if anomaly_score >= severe_threshold:
            return 1.0

        middle = (anomaly_score - anomaly_threshold) / (severe_threshold - anomaly_threshold)
        return min(1.0, 0.70 + (0.30 * middle))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _classify_risk_level(self, score: float) -> str:
        if score < self.config.low_threshold:
            return "LOW"
        if score < self.config.medium_threshold:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _detect_dominant_driver(profile_norm: float, anomaly_norm: float, forecast_norm: float) -> str:
        parts = {
            "profile": profile_norm,
            "anomaly": anomaly_norm,
            "forecast": forecast_norm,
        }
        return max(parts, key=parts.get)

    def _build_explanation(
        self,
        risk_level: str,
        profile_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        forecast_result: Dict[str, Any],
    ) -> List[str]:
        messages: List[str] = []

        messages.append(
            f"Profil sinyali: {profile_result['profile_label']} "
            f"(cluster={profile_result['cluster_id']}, norm={profile_result['profile_risk_norm']})."
        )

        if anomaly_result["is_severe_anomaly"]:
            messages.append(
                f"Son hafta ciddi anomali içeriyor "
                f"(score={anomaly_result['anomaly_score']}, severe_threshold={anomaly_result['severe_anomaly_threshold']})."
            )
        elif anomaly_result["is_anomaly"]:
            messages.append(
                f"Son hafta anomali tespit edildi "
                f"(score={anomaly_result['anomaly_score']}, threshold={anomaly_result['anomaly_threshold']})."
            )
        else:
            messages.append(
                f"Son hafta anomali baskısı düşük "
                f"(norm={anomaly_result['anomaly_risk_norm']})."
            )

        if forecast_result["budget"] > 0:
            messages.append(
                f"Ay sonu tahmini harcama={forecast_result['predicted_month_end_expense']}, "
                f"bütçe kullanım oranı={forecast_result['budget_usage_ratio']}, "
                f"budget overrun probability={forecast_result['budget_overrun_probability']}."
            )
        else:
            messages.append(
                f"Bütçe verilmediği için forecast riski harcama temposuna göre yorumlandı "
                f"(predicted_month_end_expense={forecast_result['predicted_month_end_expense']})."
            )

        messages.append(f"Nihai risk seviyesi: {risk_level}.")
        return messages


def load_profile_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Training script içindeki loader path beklediği için
    predictor tarafında DataFrame kabul eden küçük adaptör.
    """
    temp_path = _write_temp_csv(df)
    try:
        return load_profile_data(temp_path)
    finally:
        _safe_remove(temp_path)


def load_forecast_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    temp_path = _write_temp_csv(df)
    try:
        return load_forecast_data(temp_path)
    finally:
        _safe_remove(temp_path)


def _write_temp_csv(df: pd.DataFrame) -> str:
    temp_dir = os.path.join(BASE_DIR, "ml", "models_store", "_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "risk_predictor_temp_input.csv")
    df.to_csv(temp_path, index=False)
    return temp_path


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def load_anomaly_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    temp_path = _write_temp_csv(df)
    try:
        return load_anomaly_data(temp_path)
    finally:
        _safe_remove(temp_path)

if __name__ == "__main__":
    try:
        predictor = FinancialRiskPredictor()

        # örnek kullanım
        result = predictor.predict_from_csv(file_path=DEFAULT_DATA_PATH, budget=3500.0)
           
        print("\n=== FINANCIAL RISK RESULT ===")
        print(f"Risk Score   : {result['risk_score']}")
        print(f"Risk Level   : {result['risk_level']}")
        print(f"Main Driver  : {result['dominant_driver']}")

        print("\n--- Components ---")
        print(result["components"])

        print("\n--- Explanation ---")
        for item in result["explanation"]:
            print(f"- {item}")

    except Exception as exc:
        print(f"HATA: Financial risk prediction sırasında sorun oluştu: {exc}")