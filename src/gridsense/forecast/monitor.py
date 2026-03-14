"""
Model and data drift monitoring using Evidently AI.

Compares the distribution of recent inference data against the reference
training distribution to detect feature drift and model performance
degradation.  Generates HTML reports and returns a structured result dict.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DRIFT_THRESHOLD = 0.05   # p-value threshold for statistical tests


class DriftMonitor:
    """Detect data drift between reference (train) and current (recent) data.

    Parameters
    ----------
    reference_data:
        The training dataset — used as the baseline distribution.
    report_dir:
        Directory where HTML drift reports are saved.
    drift_threshold:
        p-value below which a feature is flagged as drifted.

    Example
    -------
    ::

        monitor = DriftMonitor(reference_data=train_df)
        result = monitor.check(current_data=recent_df)
        if result["drift_detected"]:
            print("Drifted features:", result["drifted_features"])
    """

    FEATURE_COLS = ["irradiance_wm2", "temp_c", "humidity_pct"]

    def __init__(
        self,
        reference_data: pd.DataFrame,
        report_dir: Path = Path("reports"),
        drift_threshold: float = _DRIFT_THRESHOLD,
    ) -> None:
        self._reference = reference_data[self.FEATURE_COLS].copy()
        self._report_dir = Path(report_dir)
        self._threshold = drift_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Run drift detection between reference and current data.

        Parameters
        ----------
        current_data:
            Recent observations to compare against the reference distribution.

        Returns
        -------
        dict with keys:
            - ``drift_detected`` (bool)
            - ``drifted_features`` (list[str])
            - ``report_path`` (str)  — path to the saved HTML report
            - ``checked_at`` (str)   — ISO timestamp
        """
        try:
            return self._check_with_evidently(current_data)
        except ImportError:
            logger.warning(
                "Evidently AI not installed — falling back to basic KS test."
            )
            return self._check_with_scipy(current_data)

    # ------------------------------------------------------------------
    # Evidently-based check
    # ------------------------------------------------------------------

    def _check_with_evidently(self, current_data: pd.DataFrame) -> dict[str, Any]:
        from evidently.metric_preset import DataDriftPreset  # type: ignore[import]
        from evidently.report import Report  # type: ignore[import]

        current = current_data[self.FEATURE_COLS].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self._reference, current_data=current)

        result_dict = report.as_dict()
        drift_metrics = result_dict["metrics"][0]["result"]

        drifted: list[str] = []
        for col in self.FEATURE_COLS:
            col_result = drift_metrics.get("drift_by_columns", {}).get(col, {})
            if col_result.get("drift_detected", False):
                drifted.append(col)

        report_path = self._save_evidently_report(report)

        return {
            "drift_detected": bool(drifted),
            "drifted_features": drifted,
            "report_path": str(report_path),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_evidently_report(self, report: Any) -> Path:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._report_dir / f"drift_report_{timestamp}.html"
        report.save_html(str(path))
        logger.info("Drift report saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Fallback: SciPy KS test
    # ------------------------------------------------------------------

    def _check_with_scipy(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Kolmogorov–Smirnov test as a lightweight Evidently fallback."""
        from scipy import stats  # type: ignore[import]

        current = current_data[self.FEATURE_COLS].copy()
        drifted: list[str] = []

        for col in self.FEATURE_COLS:
            ref_vals = self._reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            if p_value < self._threshold:
                logger.info(
                    "Drift detected in '%s': KS=%.4f, p=%.4f", col, ks_stat, p_value
                )
                drifted.append(col)

        report_path = self._save_text_report(drifted)

        return {
            "drift_detected": bool(drifted),
            "drifted_features": drifted,
            "report_path": str(report_path),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_text_report(self, drifted: list[str]) -> Path:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._report_dir / f"drift_report_{timestamp}.txt"
        with open(path, "w") as f:
            f.write(f"Drift check: {timestamp}\n")
            f.write(f"Drifted features: {drifted or 'none'}\n")
        return path
